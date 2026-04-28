#!/usr/bin/env python3
"""
为动态点云文件添加反射强度特征。

输入1 (--dyn_dir):    动态点云目录，每个PCD只有 x,y,z 三个字段（无intensity）。
                      例: /mnt/bevfusion/pcd_data/velodyne_gt_no_intensity/
输入2 (--orig_dir):   原始全场点云目录，含 x,y,z,intensity 四个字段。
                      例: /mnt/BEVHeight-main/data/dair-v2x-i/velodyne/
输出  (--out_dir):    保存带intensity动态点云的目录（binary PCD, FIELDS x y z intensity）。
                      例: /mnt/bevfusion/pcd_data/velodyne_gt/

逻辑：
  对每一帧，在原始全场点云中为每个动态点找最近邻点（kNN k=1），
  将该邻居的intensity赋给动态点。由于动态点来自同一传感器帧，
  绝大多数应存在精确/极近的匹配点。

GPU加速：
  使用 torch.cdist 在 GPU 上计算欧氏距离矩阵，支持批量处理。
"""

import os
import sys
import struct
import argparse
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch


# ─────────────────────────────────────────────
# PCD I/O helpers
# ─────────────────────────────────────────────

def read_pcd(path):
    """读取PCD文件，返回 numpy float32 数组 (N, num_fields)"""
    if os.path.getsize(path) == 0:
        warnings.warn(f"Empty PCD file: {path}")
        return np.zeros((0, 3), dtype=np.float32), {}

    meta = {}
    with open(path, 'rb') as f:
        while True:
            raw = f.readline()
            if raw == b'':
                break
            line = raw.decode('utf-8', errors='replace').strip()
            if not line or line.startswith('#'):
                continue
            key = line.split()[0].upper()
            val = line.split()[1:]
            if key == 'DATA':
                meta['data_type'] = val[0].lower()
                break
            meta[key] = val

        fields = meta.get('FIELDS', [])
        sizes = [int(s) for s in meta.get('SIZE', ['4'] * len(fields))]
        types = meta.get('TYPE', ['F'] * len(fields))
        npoints = int(meta.get('WIDTH', [0])[0])
        data_type = meta.get('data_type', 'binary')

        type_map = {('F', 4): np.float32, ('F', 8): np.float64,
                    ('U', 1): np.uint8,   ('U', 2): np.uint16,
                    ('U', 4): np.uint32,  ('U', 8): np.uint64,
                    ('I', 1): np.int8,    ('I', 2): np.int16,
                    ('I', 4): np.int32,   ('I', 8): np.int64}
        dtype = np.dtype([(f, type_map.get((t, s), np.float32))
                          for f, t, s in zip(fields, types, sizes)])

        if data_type == 'ascii':
            data = np.loadtxt(f, dtype=dtype)
        elif data_type == 'binary':
            buf = f.read(npoints * dtype.itemsize)
            data = np.frombuffer(buf, dtype=dtype)
        elif data_type == 'binary_compressed':
            import lzf
            fmt = 'II'
            comp_size, uncomp_size = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
            comp_data = f.read(comp_size)
            buf = lzf.decompress(comp_data, uncomp_size)
            data = np.zeros(npoints, dtype=dtype)
            ix = 0
            for fi, dt in enumerate(dtype.descr):
                field_name = dt[0]
                field_dtype = np.dtype(dt[1])
                nbytes = field_dtype.itemsize * npoints
                data[field_name] = np.frombuffer(buf[ix:ix+nbytes], field_dtype)
                ix += nbytes
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    output = np.zeros((len(data), len(dtype)), dtype=np.float32)
    for ic, field_name in enumerate(fields):
        output[:, ic] = data[field_name].astype(np.float32)
    return output, meta


def write_pcd_binary(path, points_xyzi):
    """将 (N,4) float32 数组写为 binary PCD 文件 (x,y,z,intensity)"""
    npoints = len(points_xyzi)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {npoints}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {npoints}\n"
        "DATA binary\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(points_xyzi.astype(np.float32).tobytes())


# ─────────────────────────────────────────────
# Core matching (GPU)
# ─────────────────────────────────────────────

def match_intensity_gpu(dyn_xyz, orig_xyzi, device):
    """
    输入:
        dyn_xyz:    (N, 3) float32  动态点 xyz
        orig_xyzi:  (M, 4) float32  原始点 xyz + intensity
    输出:
        result_xyzi:(N, 4) float32  动态点 xyz + 最近邻intensity
    """
    if len(dyn_xyz) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    dyn_t  = torch.from_numpy(dyn_xyz[:, :3]).float().to(device)     # (N,3)
    orig_t = torch.from_numpy(orig_xyzi[:, :3]).float().to(device)   # (M,3)

    # 分块计算距离（避免显存OOM）：block_size 行
    BLOCK = 4096
    nn_idx = torch.zeros(len(dyn_t), dtype=torch.long, device=device)
    for start in range(0, len(dyn_t), BLOCK):
        end = min(start + BLOCK, len(dyn_t))
        dists = torch.cdist(dyn_t[start:end], orig_t)   # (block, M)
        nn_idx[start:end] = dists.argmin(dim=1)

    intensities = torch.from_numpy(orig_xyzi[:, 3]).float().to(device)[nn_idx]
    intensities = intensities.cpu().numpy()

    result = np.concatenate([dyn_xyz[:, :3], intensities[:, np.newaxis]], axis=1)
    return result.astype(np.float32)


# ─────────────────────────────────────────────
# Per-file worker
# ─────────────────────────────────────────────

def process_one(fname, dyn_dir, orig_dir, out_dir, device):
    dyn_path  = os.path.join(dyn_dir, fname)
    orig_path = os.path.join(orig_dir, fname)
    out_path  = os.path.join(out_dir, fname)

    try:
        dyn_pts, _  = read_pcd(dyn_path)   # (N,3)
        orig_pts, _ = read_pcd(orig_path)  # (M,4)

        if len(dyn_pts) == 0:
            # 空帧：写一个dummy点
            write_pcd_binary(out_path, np.array([[0.001, 0.0, 0.0, 0.0]], dtype=np.float32))
            return fname, 'empty'

        result = match_intensity_gpu(dyn_pts, orig_pts, device)
        write_pcd_binary(out_path, result)
        return fname, 'ok'

    except Exception as e:
        return fname, f'ERROR: {e}'


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Add intensity to dynamic PCD files using GPU kNN matching.")
    parser.add_argument('--dyn_dir',  required=True, help='Input: dynamic XYZ-only PCD directory')
    parser.add_argument('--orig_dir', required=True, help='Input: original XYZ+intensity PCD directory')
    parser.add_argument('--out_dir',  required=True, help='Output: XYZ+intensity dynamic PCD directory')
    parser.add_argument('--workers',  type=int, default=4,  help='Parallel IO workers (default 4)')
    parser.add_argument('--gpu',      type=int, default=0,  help='GPU device index (default 0)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_files = sorted(os.listdir(args.dyn_dir))
    pcd_files = [f for f in all_files if f.endswith('.pcd')]
    print(f"Found {len(pcd_files)} PCD files in {args.dyn_dir}")

    errors = []
    with tqdm(total=len(pcd_files), desc="Processing frames", unit="frame") as pbar:
        # 串行处理（GPU kNN 本身已充分利用GPU；IO用多线程）
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, fname, args.dyn_dir, args.orig_dir, args.out_dir, device): fname
                for fname in pcd_files
            }
            for fut in as_completed(futures):
                fname, status = fut.result()
                if 'ERROR' in str(status):
                    errors.append((fname, status))
                pbar.update(1)

    print(f"\nDone. Processed {len(pcd_files)} files.")
    print(f"Errors: {len(errors)}")
    for fname, err in errors[:10]:
        print(f"  {fname}: {err}")

    # 快速验证一个输出文件
    sample = pcd_files[0]
    out_path = os.path.join(args.out_dir, sample)
    if os.path.exists(out_path):
        pts, meta = read_pcd(out_path)
        print(f"\nSample output [{sample}]: shape={pts.shape}, fields={meta.get('FIELDS')}")


if __name__ == '__main__':
    main()
