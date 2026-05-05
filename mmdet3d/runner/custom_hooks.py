"""
自定义训练Hook:
  1. GradientExplosionHook  — 每次iter后检查梯度范数，发现梯度爆炸则停止训练并写失败报告
     注意：FP16训练时Fp16OptimizerHook会自动跳过NaN梯度的更新步骤（正常行为），
     因此需要容忍一定次数的NaN/Inf，连续超过nan_tolerance次才视为真正爆炸。
  2. PlateauEarlyStopHook   — 跟踪周期性评估的mAP，连续两次评估相差极小则停止训练
"""

import os
import math
import traceback
from datetime import datetime

import torch
from mmcv.runner import HOOKS, Hook, Priority, Priority


# ─────────────────────────────────────────────────────────────────────────────
# 1. 梯度爆炸检测 Hook
# ─────────────────────────────────────────────────────────────────────────────
@HOOKS.register_module()
class GradientExplosionHook(Hook):
    """检查每个 iteration 后的梯度范数，若满足爆炸条件则写失败报告并终止训练。

    FP16 说明：
      Fp16OptimizerHook 会自动处理溢出（跳过本次更新），梯度 NaN/Inf 是初期正常
      现象。本 Hook 通过 nan_tolerance 参数设置连续多少次 NaN 才触发停训。

    Args:
        max_grad_norm (float): 梯度 L2 范数阈值（非NaN情形），默认 500.0。
        nan_tolerance (int): 允许连续出现 NaN/Inf 梯度的次数上限，默认 20。
        fail_report_dir (str): 失败报告保存目录，默认使用 work_dir。
    """

    def __init__(
        self,
        max_grad_norm: float = 500.0,
        nan_tolerance: int = 20,
        fail_report_dir: str = None,
    ):
        self.max_grad_norm = max_grad_norm
        self.nan_tolerance = nan_tolerance
        self.fail_report_dir = fail_report_dir
        self._consecutive_nan = 0   # 连续 NaN 计数
        self._consecutive_large = 0  # 连续大范数计数

    def after_train_iter(self, runner):
        total_norm = 0.0
        has_grad = False
        has_nan = False
        try:
            for p in runner.model.parameters():
                if p.grad is not None:
                    has_grad = True
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan = True
                        break
                    param_norm = p.grad.data.norm(2).item()
                    if math.isnan(param_norm) or math.isinf(param_norm):
                        has_nan = True
                        break
                    total_norm += param_norm ** 2

            if not has_grad:
                return

            if has_nan:
                self._consecutive_nan += 1
                self._consecutive_large = 0
                if self._consecutive_nan >= self.nan_tolerance:
                    self._stop(
                        runner,
                        f"NaN/Inf gradient persisted for {self._consecutive_nan} "
                        f"consecutive iters (tolerance={self.nan_tolerance}). "
                        f"Epoch={runner.epoch+1}, Iter={runner.iter+1}",
                    )
                else:
                    runner.logger.warning(
                        f"[GradientExplosionHook] NaN/Inf gradient at "
                        f"epoch={runner.epoch+1} iter={runner.iter+1} "
                        f"(count={self._consecutive_nan}/{self.nan_tolerance}, "
                        f"FP16 overflow is normal at training start)"
                    )
                return

            # 梯度正常，重置 NaN 计数
            self._consecutive_nan = 0
            total_norm = math.sqrt(total_norm)
            if total_norm > self.max_grad_norm:
                self._consecutive_large += 1
                runner.logger.warning(
                    f"[GradientExplosionHook] Large gradient norm={total_norm:.1f} "
                    f"at epoch={runner.epoch+1} iter={runner.iter+1} "
                    f"(count={self._consecutive_large})"
                )
                if self._consecutive_large >= 5:
                    self._stop(
                        runner,
                        f"Gradient norm={total_norm:.2f} > threshold={self.max_grad_norm} "
                        f"for {self._consecutive_large} consecutive iters",
                    )
            else:
                self._consecutive_large = 0

        except Exception as e:
            runner.logger.warning(f"[GradientExplosionHook] exception: {e}")

    def _stop(self, runner, reason: str):
        runner.logger.error(f"[GradientExplosionHook] STOPPING training — {reason}")
        report_dir = self.fail_report_dir or runner.work_dir
        os.makedirs(report_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"training_failed_{ts}.txt")
        with open(report_path, "w") as f:
            f.write("Training failure report\n")
            f.write(f"Time       : {ts}\n")
            f.write(f"Epoch      : {runner.epoch + 1}\n")
            f.write(f"Iteration  : {runner.iter + 1}\n")
            f.write(f"Reason     : {reason}\n")
        runner.logger.error(
            f"[GradientExplosionHook] Failure report saved to {report_path}"
        )
        runner._max_epochs = runner.epoch


# ─────────────────────────────────────────────────────────────────────────────
# 2. 平台期早停 Hook
# ─────────────────────────────────────────────────────────────────────────────
@HOOKS.register_module()
class PlateauEarlyStopHook(Hook):
    """在 DistEvalHook 完成评估后，从 runner.log_buffer 读取 mAP，
    若连续 patience 次评估之间变化量 < min_delta，则终止训练。

    必须在 DistEvalHook **之后** 注册（priority 更低）。

    Args:
        metric_key (str)  : log_buffer 中记录 mAP 的 key，默认 'mAP_3d_moderate'。
        min_delta (float) : 被认为"显著改善"的最小变化量，默认 0.005（0.5%）。
        patience (int)    : 连续多少次无改善则停止，默认 2。
        start_epoch (int) : 最早在哪个 epoch 开始计数，默认 30。
        eval_interval (int): 评估间隔（与 DistEvalHook 的 interval 保持一致），默认 10。
    """

    def __init__(
        self,
        metric_key: str = "mAP_3d_moderate",
        min_delta: float = 0.005,
        patience: int = 2,
        start_epoch: int = 30,
        eval_interval: int = 10,
    ):
        self.metric_key = metric_key
        self.min_delta = min_delta
        self.patience = patience
        self.start_epoch = start_epoch
        self.eval_interval = eval_interval
        self._history = []
        self._no_improve_count = 0

    def _is_eval_epoch(self, runner) -> bool:
        epoch = runner.epoch
        if epoch < self.start_epoch:
            return False
        return (epoch - self.start_epoch) % self.eval_interval == 0

    def after_train_epoch(self, runner):
        if not self._is_eval_epoch(runner):
            return

        # PRIMARY: 从 JSON 文件读取（因为本 Hook priority=VERY_LOW，运行时 log_buffer 已被 TextLoggerHook 清除）
        import json as _json, os as _os
        val = None
        _json_path = _os.path.join(runner.work_dir, 'outputs', 'last_eval_result.json')
        try:
            if _os.path.exists(_json_path):
                with open(_json_path) as _f:
                    _data = _json.load(_f)
                val = _data.get(self.metric_key)
                if val is not None:
                    runner.logger.info(
                        f"[PlateauEarlyStopHook] read {self.metric_key}={val:.4f} from eval result JSON"
                    )
        except Exception as _e:
            runner.logger.warning(f"[PlateauEarlyStopHook] failed to read JSON: {_e}")

        # FALLBACK: 尝试 log_buffer（如果 JSON 读取失败）
        if val is None:
            if hasattr(runner, "log_buffer") and runner.log_buffer.output:
                val = runner.log_buffer.output.get(self.metric_key, None)

        if val is None:
            runner.logger.info(
                f"[PlateauEarlyStopHook] metric '{self.metric_key}' not found "
                f"in log_buffer or fallback file at epoch {runner.epoch}, skip plateau check."
            )
            return

        runner.logger.info(
            f"[PlateauEarlyStopHook] epoch={runner.epoch}, "
            f"{self.metric_key}={val:.4f}, history={[f'{v:.4f}' for v in self._history]}"
        )

        if len(self._history) > 0:
            last_val = self._history[-1]
            delta = abs(val - last_val)
            if delta < self.min_delta:
                self._no_improve_count += 1
                runner.logger.info(
                    f"[PlateauEarlyStopHook] No improvement: delta={delta:.4f} < "
                    f"min_delta={self.min_delta}, count={self._no_improve_count}/{self.patience}"
                )
                if self._no_improve_count >= self.patience:
                    runner.logger.info(
                        f"[PlateauEarlyStopHook] Plateau detected! "
                        f"Stopping training at epoch {runner.epoch}."
                    )
                    self._write_plateau_report(runner, val, last_val)
                    runner._max_epochs = runner.epoch
                    return
            else:
                self._no_improve_count = 0
        else:
            self._no_improve_count = 0

        self._history.append(val)

    def _write_plateau_report(self, runner, current_val, last_val):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(runner.work_dir, f"plateau_early_stop_{ts}.txt")
        with open(report_path, "w") as f:
            f.write("Plateau Early Stop Report\n")
            f.write(f"Time            : {ts}\n")
            f.write(f"Stopped epoch   : {runner.epoch}\n")
            f.write(f"metric_key      : {self.metric_key}\n")
            f.write(f"current value   : {current_val:.4f}\n")
            f.write(f"last value      : {last_val:.4f}\n")
            f.write(f"delta           : {abs(current_val - last_val):.4f}\n")
            f.write(f"min_delta       : {self.min_delta}\n")
            f.write(f"Full history    : {self._history}\n")
        runner.logger.info(f"[PlateauEarlyStopHook] Report saved to {report_path}")
