#!/usr/bin/env python3
"""单GPU显存测试：逐步增加batch size，找到上限"""
import sys, os, gc, time
import torch
from mmcv import Config

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else 'experiments/Exp07_two_stream/config.yaml'
START_BS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.v2x_dataset import collate_fn
from functools import partial

torch.backends.cudnn.benchmark = True

cfg = Config.fromfile(CONFIG_PATH)
cfg.model.train_cfg = None

model = build_model(cfg.model).cuda()
model.train()
print(f'Model built. Params: {sum(p.numel() for p in model.parameters()):,}')

dataset = build_dataset(cfg.data.train)
cfg_bs = cfg.data.samples_per_gpu
print(f'Config BS={cfg_bs}, testing BS from {START_BS}...')

runner = None  # will be set after successful forward

for bs in range(START_BS, 33, 2):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        samples = [dataset[i] for i in range(min(bs, len(dataset)))]
        if len(samples) < bs:
            # wrap around
            samples = samples * (bs // len(samples) + 1)
            samples = samples[:bs]

        collated = [list(s) for s in zip(*samples)]
        batch = partial(collate_fn, is_return_depth=False)(collated)

        img = batch['img'].cuda()
        points = [p.cuda() for p in batch['points']]
        points_bg = [p.cuda() for p in batch.get('points_bg', batch['points'])]
        metas = batch['metas']

        with torch.cuda.amp.autocast():
            output = model(
                img=img, points=points,
                camera2ego=batch['camera2ego'].cuda(),
                lidar2ego=batch['lidar2ego'].cuda(),
                lidar2camera=batch['lidar2camera'].cuda(),
                lidar2image=batch['lidar2image'].cuda(),
                camera_intrinsics=batch['camera_intrinsics'].cuda(),
                camera2lidar=batch['camera2lidar'].cuda(),
                img_aug_matrix=batch['img_aug_matrix'].cuda(),
                lidar_aug_matrix=batch['lidar_aug_matrix'].cuda(),
                metas=metas, points_bg=points_bg,
            )

        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        total_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3

        print(f'  BS={bs:2d}: ✅ peak {peak_mb:.0f} MB / {total_gb:.1f} GB ({peak_mb/1024/total_gb*100:.0f}%)')

        if peak_mb / 1024 > total_gb * 0.88:
            print(f'  ⚠️  BS={bs} at {peak_mb/1024/total_gb*100:.0f}% — near limit, stop testing')
            break

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'  BS={bs:2d}: ❌ OOM')
        else:
            print(f'  BS={bs:2d}: ❌ {type(e).__name__}: {str(e)[:80]}')
        break
    except Exception as e:
        print(f'  BS={bs:2d}: ❌ {type(e).__name__}: {str(e)[:80]}')
        break

print('\nDone. Recommended samples_per_gpu is the last ✅ value.')
