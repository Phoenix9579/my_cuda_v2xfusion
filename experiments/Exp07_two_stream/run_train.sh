#!/bin/bash
# ============================================================
# Exp07 — Two-Stream PointPillars（双流架构）
# 前景点云 + 背景点云 + 相机 → 门控融合 → 3D检测
# 用法: bash /mnt/bevfusion/experiments/Exp07_two_stream/run_train.sh
# ============================================================
set -e

WORKDIR=/mnt/bevfusion
EXP_DIR=${WORKDIR}/experiments/Exp07_two_stream
CONFIG=${EXP_DIR}/config.yaml
PYTHON=/root/miniconda3/envs/v2xfusion/bin/python
TORCHPACK=/root/miniconda3/envs/v2xfusion/bin/torchpack
NUM_GPU=16
LOG_FILE=${EXP_DIR}/train_launch.log

mkdir -p ${EXP_DIR}/outputs
cd ${WORKDIR}

# 断点续训
LATEST_CKPT=$(ls ${EXP_DIR}/epoch_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -n "${LATEST_CKPT}" ]; then
    EPOCH_NUM=$(basename ${LATEST_CKPT} .pth | sed 's/epoch_//')
    echo "检测到最新 checkpoint: ${LATEST_CKPT}（epoch ${EPOCH_NUM}），将从此处续训" | tee -a ${LOG_FILE}
    sed -i "s|^resume_from:.*|resume_from: ${LATEST_CKPT}|" ${CONFIG}
else
    echo "未找到已有 checkpoint，从头开始训练" | tee -a ${LOG_FILE}
    sed -i "s|^resume_from:.*|resume_from: null|" ${CONFIG}
fi

echo "=====================================================" | tee -a ${LOG_FILE}
echo "Exp07 Two-Stream 训练启动: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${LOG_FILE}
echo "Config  : ${CONFIG}" | tee -a ${LOG_FILE}
echo "GPU数   : ${NUM_GPU}" | tee -a ${LOG_FILE}
echo "Run dir : ${EXP_DIR}" | tee -a ${LOG_FILE}
echo "=====================================================" | tee -a ${LOG_FILE}

${TORCHPACK} dist-run \
    -np ${NUM_GPU} \
    ${PYTHON} scripts/train.py \
    ${CONFIG} \
    --mode dense \
    --run-dir ${EXP_DIR} \
    2>&1 | tee -a ${EXP_DIR}/train_stdout.log

EXIT_CODE=${PIPESTATUS[0]}
echo "" | tee -a ${LOG_FILE}
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[SUCCESS] 训练正常结束。$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${LOG_FILE}
else
    echo "[FAILED]  训练异常退出，exit code=${EXIT_CODE}。$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${LOG_FILE}
    FAIL_REPORTS=$(ls ${EXP_DIR}/training_failed_*.txt 2>/dev/null | wc -l)
    if [ "${FAIL_REPORTS}" -eq "0" ]; then
        TS=$(date '+%Y%m%d_%H%M%S')
        REPORT=${EXP_DIR}/training_failed_${TS}.txt
        echo "Training failure report" > ${REPORT}
        echo "Time      : ${TS}" >> ${REPORT}
        echo "Exit code : ${EXIT_CODE}" >> ${REPORT}
        echo "Log       : ${EXP_DIR}/train_stdout.log" >> ${REPORT}
        echo "[FAILED] Failure report saved to ${REPORT}" | tee -a ${LOG_FILE}
    fi
    exit ${EXIT_CODE}
fi
