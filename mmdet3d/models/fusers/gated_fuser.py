"""GatedFuser: Two-Stream LiDAR + Camera gated fusion for roadside perception.

Architecture:
    Foreground LiDAR BEV [B, C_fg, H, W]  ─┐
    Background LiDAR BEV [B, C_bg, H, W]  ─┤
    Camera BEV          [B, C_cam, H, W]  ─┤
                                             ├─→ Gate ─→ [B, 1, H, W] 逐位置权重
                                             ├─→ Fuse ─→ [B, C_out, H, W]
                                             └─→ Residual ─→ output

The gate learns WHERE to trust LiDAR (fg+bg) vs Camera at each BEV location.
"""
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["GatedFuser"]


@FUSERS.register_module()
class GatedFuser(nn.Module):
    """Gated fusion with learned per-location weighting between LiDAR and camera.

    Args:
        in_channels (List[int]): [C_fg, C_bg, C_cam] input channels.
        out_channels (int): Output channel count.
    """

    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        total_in = sum(in_channels)

        # Gate: learn a spatial attention map
        self.gate = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Fuse: combine all modalities
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

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

        # Spatial gate: where to trust LiDAR vs camera
        gate = self.gate(cat)          # [B, 1, H, W]
        fused = self.fuse(cat)         # [B, C_out, H, W]

        # Residual: gate * fused_LiDAR + (1-gate) * camera
        return gate * fused + (1 - gate) * cam_bev
