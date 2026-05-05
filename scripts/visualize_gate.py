#!/usr/bin/env python3
"""
可视化 GatedFuser 的门控热力图 (gate heatmap)。
用于论文中展示模型在哪些BEV位置信任LiDAR vs 相机。

用法:
    python scripts/visualize_gate.py \
        experiments/Exp07_two_stream/config.yaml \
        experiments/Exp07_two_stream/epoch_40.pth \
        --sample-idx 0 \
        --out-dir experiments/Exp07_two_stream/gate_viz/

输出:
    gate_<sample_idx>.png  — gate热力图 + GT框叠加 + 相机图缩略图
"""
import argparse
import os
import numpy as np
import torch
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.datasets.v2x_dataset import collate_fn
from functools import partial
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint path')
    parser.add_argument('--sample-idx', type=int, default=0, help='which sample to visualize')
    parser.add_argument('--out-dir', default=None, help='output directory')
    parser.add_argument('--num-samples', type=int, default=4, help='number of samples to process')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.checkpoint), 'gate_viz')
    os.makedirs(args.out_dir, exist_ok=True)

    # Load config and build model
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None  # eval mode
    model = build_model(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    model.eval()

    # Build dataset
    dataset = build_dataset(cfg.data.test)
    dataloader_collate = partial(collate_fn, is_return_depth=False)

    print(f'Dataset size: {len(dataset)}')
    print(f'Processing {args.num_samples} samples...')

    for idx in range(args.sample_idx, min(args.sample_idx + args.num_samples, len(dataset))):
        sample = dataset[idx]

        # Batch single sample
        batched = [list(s) for s in zip(*[sample])]
        batch_dict = dataloader_collate(batched)

        img = batch_dict['img'].cuda()
        points = [p.cuda() for p in batch_dict['points']]
        points_bg = [p.cuda() for p in batch_dict.get('points_bg', batch_dict['points'])]
        metas = batch_dict['metas']

        # Forward with gate extraction
        with torch.no_grad():
            # We need to modify forward to return gate
            # For now, we monkey-patch the fuser to save gate
            gate_value = [None]
            original_forward = model.fuser.forward

            def patched_forward(inputs):
                fg_bev, bg_bev, cam_bev = inputs
                cat = torch.cat([fg_bev, bg_bev, cam_bev], dim=1)
                g = model.fuser.gate(cat)
                gate_value[0] = g.cpu().numpy()
                fused = model.fuser.fuse(cat)
                fg_cam = torch.cat([fg_bev, cam_bev], dim=1)
                degraded = model.fuser.degrad_fuser(fg_cam)
                return g * fused + (1 - g) * degraded

            model.fuser.forward = patched_forward

            _ = model(
                img=img,
                points=points,
                camera2ego=batch_dict['camera2ego'].cuda(),
                lidar2ego=batch_dict['lidar2ego'].cuda(),
                lidar2camera=batch_dict['lidar2camera'].cuda(),
                lidar2image=batch_dict['lidar2image'].cuda(),
                camera_intrinsics=batch_dict['camera_intrinsics'].cuda(),
                camera2lidar=batch_dict['camera2lidar'].cuda(),
                img_aug_matrix=batch_dict['img_aug_matrix'].cuda(),
                lidar_aug_matrix=batch_dict['lidar_aug_matrix'].cuda(),
                metas=metas,
                points_bg=points_bg,
            )

            model.fuser.forward = original_forward  # restore

        gate = gate_value[0]  # [B, 1, 128, 128]
        if gate is None:
            print(f'  Sample {idx}: gate extraction failed, skipping')
            continue

        # ── Render ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Gate heatmap
        gate_map = gate[0, 0]  # [128, 128]
        im = axes[0].imshow(gate_map, cmap='coolwarm', vmin=0, vmax=1, origin='lower',
                            extent=[0, 102.4, -51.2, 51.2])
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title(f'Gate Heatmap (sample {idx})')
        plt.colorbar(im, ax=axes[0], label='Gate value (1=trust LiDAR, 0=trust Foreground-only)')

        # Panel 2: Camera image thumbnail
        try:
            # img is [B, N, C, H, W] — show first camera
            cam_img = img[0, 0].cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            cam_img = cam_img * std + mean
            cam_img = cam_img.astype(np.uint8)
            axes[1].imshow(cam_img)
            axes[1].set_title('Camera Image')
            axes[1].axis('off')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Camera image unavailable\n{e}', transform=axes[1].transAxes, ha='center')
            axes[1].axis('off')

        out_path = os.path.join(args.out_dir, f'gate_{idx:04d}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f'  Sample {idx}: saved to {out_path}')

    print(f'Done. {args.num_samples} samples visualized to {args.out_dir}')


if __name__ == '__main__':
    main()
