"""GatedFuser: Two-Stream LiDAR + Camera gated fusion for roadside perception.

Architecture:
    Foreground LiDAR BEV [B, C_fg, H, W]  ─┐
    Background LiDAR BEV [B, C_bg, H, W]  ─┤
    Camera BEV          [B, C_cam, H, W]  ─┤
                                             ├─→ Gate ─→ [B, 1, H, W] 逐位置权重
                                             ├─→ Fuse ─→ [B, C_out, H, W]
                                             └─→ Residual ─→ output

Residual degradation: when gate→0, output → fg+cam fused (≈ Exp06 foreground-only).

Stability fixes (2026-05-05):
  - Gate final conv: add bias=True, init to 0 → sigmoid(0)=0.5 as starting point
  - Gate first conv: kaiming init with small gain to prevent large initial activations
  - Weight init: all conv layers properly initialized with kaiming_uniform_
"""
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["GatedFuser"]


@FUSERS.register_module()
class GatedFuser(nn.Module):
    """Gated fusion: fg + bg LiDAR + camera with Exp06 foreground-only degradation.

    Args:
        in_channels (List[int]): [C_fg, C_bg, C_cam] input channels.
        out_channels (int): Output channel count.
    """

    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        total_in = sum(in_channels)

        # Gate: learn a spatial attention map [B, 1, H, W]
        #   - bias=True on final conv, initialized to 0 → sigmoid(0)=0.5 at start
        #   - keep 3×3 for checkpoint compatibility with epoch_30.pth
        self.gate = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Fuse: three-stream fusion (fg + bg + cam)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Degradation path: when gate→0, fall back to Exp06 (fg + cam only)
        # 1×1 conv aligns fg(64ch)+cam(80ch) → out_channels(80ch)
        self.degrad_fuser = nn.Sequential(
            nn.Conv2d(in_channels[0] + in_channels[2], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization for training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming init with small nonlinearity gain for gate, normal for fuse
                nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: [fg_bev, bg_bev, cam_bev] — three BEV feature maps.

        Returns:
            torch.Tensor: Fused BEV feature [B, out_channels, H, W].
        """
        assert len(inputs) == 3, f"GatedFuser expects 3 inputs (fg, bg, cam), got {len(inputs)}"

        fg_bev, bg_bev, cam_bev = inputs
        cat = torch.cat([fg_bev, bg_bev, cam_bev], dim=1)

        # Spatial gate: where to trust LiDAR (fg+bg) vs foreground-only (Exp06)
        gate = self.gate(cat)          # [B, 1, H, W]
        fused = self.fuse(cat)         # [B, C_out, H, W]  three-stream

        # Degradation to Exp06: fg + cam → ConvFuser-equivalent
        fg_cam = torch.cat([fg_bev, cam_bev], dim=1)
        degraded = self.degrad_fuser(fg_cam)  # [B, C_out, H, W]  two-stream

        # gate * three_stream + (1-gate) * exp06_foreground_only
        return gate * fused + (1 - gate) * degraded
