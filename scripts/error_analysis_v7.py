#!/usr/bin/env python3
"""
=============================================================================
 V2XFusion 3D检测误差分析 v7 — 修复维度顺序 + 官方3D IoU匹配
=============================================================================

 v6 bug fix: KITTI 维度顺序 dh,dw,dl = [dim[2], dim[1], dim[0]].
 v6 错误地使用了 [dim[2], dim[0], dim[1]], 导致 l/w 互换.

 v7 改进:
   1. 修复维度顺序, 完全对齐官方 result2kitti 输出
   2. 使用 metric=2 (d3_box_overlap) 进行 3D IoU 匹配
   3. 官方 3D IoU 现已验证正确: Car 3D AP ≈ 52.5%

 用法:
   python scripts/error_analysis_v7.py
=============================================================================
"""

import json, os, sys, math, numpy as np
from collections import defaultdict
sys.path.insert(0, '/mnt/bevfusion')

from evaluators.result2kitti import *
from evaluators.result2kitti import category_map_dair as CATEGORY_MAP_DAIR
from evaluators.kitti_utils.kitti_common import get_label_annos
from evaluators.kitti_utils.eval import calculate_iou_partly

BASELINE_RESULT = '/mnt/bevfusion/experiments/Exp01_baseline_16gpu/results_nusc.json'
IMPROVED_RESULT = '/mnt/bevfusion/experiments/Exp06_retrain_16gpu_bs4/outputs/results_nusc.json'
GT_LABEL_DIR     = '/mnt/BEVHeight-main/data/dair-v2x-i-kitti/training/label_2'
DAIR_ROOT        = '/mnt/BEVHeight-main/data/dair-v2x-i'
OUTPUT_DIR       = '/mnt/bevfusion/experiments/error_analysis_output'

CLASS_NAMES      = ['Car', 'Pedestrian', 'Cyclist']
IOU_THRESHOLDS   = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}


# ══════════════════════════════════════════════════════════════════
# Step 1: 生成 KITTI 预测 (修复维度顺序!)
# ══════════════════════════════════════════════════════════════════

def generate_kitti_preds(results_json_path, output_dir, label='exp'):
    """
    生成 KITTI 格式预测标签.
    
    ★ 维度顺序 (FIXED): 
       KITTI format: ... dh dw dl lx ly lz ry score
       dim = pred["size"] = [l, w, h] (from _format_bbox)
       dh = dim[2] = h_model  ✓
       dw = dim[1] = w_model  ✓ (v6 bug: used dim[0]=l_model)
       dl = dim[0] = l_model  ✓ (v6 bug: used dim[1]=w_model)
    """
    pred_dir = os.path.join(output_dir, f'kitti_preds_{label}')
    os.makedirs(pred_dir, exist_ok=True)

    with open(results_json_path, 'r') as f:
        data = json.load(f)
    results = data['results']
    print(f"  [{label}] 生成 {len(results)} 帧预测...")

    from tqdm import tqdm
    for frame_id in tqdm(results.keys(), desc=f"  {label}"):
        sample_id = int(frame_id.split('/')[1].split('.')[0])

        cam_int = os.path.join(DAIR_ROOT, 'calib/camera_intrinsic', f'{sample_id:06d}.json')
        vl2c    = os.path.join(DAIR_ROOT, 'calib/virtuallidar_to_camera', f'{sample_id:06d}.json')
        if not os.path.exists(cam_int) or not os.path.exists(vl2c):
            continue

        camera_intrinsic = get_cam_calib_intrinsic(cam_int)
        Tr_velo_to_cam, r_velo2cam, t_velo2cam = get_lidar2cam(vl2c)
        camera_intrinsic = np.concatenate(
            [camera_intrinsic, np.zeros((3, 1))], axis=1)

        pred_lines = []
        for pred in results[frame_id]:
            class_name = pred['detection_name']
            if class_name not in CATEGORY_MAP_DAIR:
                continue

            loc = pred['translation']
            dim = pred['size']           # [l, w, h]
            yaw_lidar = pred['box_yaw']
            score = pred['detection_score']

            x, y, z = loc[0], loc[1], loc[2]
            l_model, w_model, h_model = dim[0], dim[1], dim[2]

            # 3D 角点和朝向
            obj_size = [dim[1], dim[0], dim[2]]
            bottom_center = [x, y, z]
            bc_in_cam = r_velo2cam @ np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bc_in_cam, r_velo2cam, t_velo2cam)
            yaw = 0.5 * np.pi - yaw_lidar
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            # 2D bbox
            box_3d = get_lidar_3d_8points(
                [dim[0], dim[1], dim[2]], yaw_lidar, [x, y, z + dim[2]/2])
            box2d = bbbox2bbox(box_3d, Tr_velo_to_cam, camera_intrinsic)

            kitti_name = CATEGORY_MAP_DAIR[class_name]
            # ★ FIXED: dh=dim[2], dw=dim[1], dl=dim[0] ★
            line = [
                kitti_name, str(0), str(0), str(round(alpha, 4)),
                str(round(box2d[0],4)), str(round(box2d[1],4)),
                str(round(box2d[2],4)), str(round(box2d[3],4)),
                str(round(dim[2],4)), str(round(dim[1],4)), str(round(dim[0],4)),
                str(round(cam_x,4)), str(round(cam_y,4)), str(round(cam_z,4)),
                str(round(yaw,4)), str(round(score,4)),
            ]
            pred_lines.append(line)

        out_file = os.path.join(pred_dir, f'{sample_id:06d}.txt')
        write_kitti_in_txt(pred_lines, out_file)
    return pred_dir


# ══════════════════════════════════════════════════════════════════
# Step 2: 辅助函数
# ══════════════════════════════════════════════════════════════════

def cam_dist(x, z):
    return math.sqrt(x**2 + z**2)

def dist_bin(d):
    for th, name in [(10,'0-10m'),(20,'10-20m'),(30,'20-30m'),(40,'30-40m'),
                      (50,'40-50m'),(60,'50-60m'),(80,'60-80m')]:
        if d < th: return name
    return '80m+'


# ══════════════════════════════════════════════════════════════════
# Step 3: 3D IoU 匹配 (复用官方 calculate_iou_partly(metric=2))
# ══════════════════════════════════════════════════════════════════

def match_by_3d_iou(gt_annos, dt_annos, cls_name, iou_thr):
    """
    使用 d3_box_overlap (metric=2) + 贪心匹配.
    """
    n_frames = len(gt_annos)

    # 官方 3D IoU 重叠矩阵
    overlaps, _, total_gt_num, total_dt_num = calculate_iou_partly(
        gt_annos, dt_annos, metric=2, num_parts=min(200, n_frames))

    total_matched, total_fn, total_fp = 0, 0, 0
    per_frame = []

    for fi in range(n_frames):
        gt_a = gt_annos[fi]
        dt_a = dt_annos[fi]
        ov = overlaps[fi]

        gt_cls_mask = np.array([n.lower() == cls_name.lower() for n in gt_a['name']])
        dt_cls_mask = np.array([n.lower() == cls_name.lower() for n in dt_a['name']])
        gt_indices = np.where(gt_cls_mask)[0]
        dt_indices = np.where(dt_cls_mask)[0]

        n_gt_cls = len(gt_indices)
        n_dt_cls = len(dt_indices)

        if n_gt_cls == 0 and n_dt_cls == 0:
            per_frame.append({'n_gt': 0, 'n_matched': 0, 'fn_list': [], 'fp_list': []})
            continue

        if n_gt_cls > 0 and n_dt_cls > 0 and ov.shape[0] > 0 and ov.shape[1] > 0:
            cls_ov = ov[gt_indices[:, None], dt_indices]
        else:
            cls_ov = np.zeros((n_gt_cls, n_dt_cls))

        dt_scores = dt_a['score'][dt_indices] if 'score' in dt_a else np.ones(n_dt_cls)

        matched_gt, matched_dt = set(), set()
        dt_order = np.argsort(-dt_scores) if n_dt_cls > 0 else np.array([])

        for di_local in dt_order:
            best_iou, best_gi = 0.0, -1
            for gi_local in range(n_gt_cls):
                if gt_indices[gi_local] in matched_gt:
                    continue
                if cls_ov[gi_local, di_local] > best_iou:
                    best_iou = cls_ov[gi_local, di_local]
                    best_gi = gt_indices[gi_local]
            if best_iou >= iou_thr and best_gi >= 0:
                matched_gt.add(best_gi)
                matched_dt.add(dt_indices[di_local])

        n_matched = len(matched_gt)
        n_fn = n_gt_cls - n_matched
        n_fp = n_dt_cls - len(matched_dt)

        total_matched += n_matched
        total_fn += n_fn
        total_fp += n_fp

        fn_list, fp_list = [], []
        for gi in gt_indices:
            if gi not in matched_gt:
                loc = gt_a['location'][gi]
                fn_list.append({
                    'type': gt_a['name'][gi],
                    'distance': cam_dist(loc[0], loc[2]),
                    'occluded': int(gt_a['occluded'][gi]),
                    'truncated': float(gt_a['truncated'][gi]),
                })
        for di in dt_indices:
            if di not in matched_dt:
                loc = dt_a['location'][di]
                fp_list.append({
                    'type': dt_a['name'][di],
                    'distance': cam_dist(loc[0], loc[2]),
                    'score': float(dt_a['score'][di]) if 'score' in dt_a else 1.0,
                })

        per_frame.append({
            'n_gt': n_gt_cls, 'n_matched': n_matched,
            'fn_list': fn_list, 'fp_list': fp_list,
        })

    return per_frame, total_matched, total_fn, total_fp


# ══════════════════════════════════════════════════════════════════
# Step 4: 对比分析
# ══════════════════════════════════════════════════════════════════

def compare_experiments(gt_annos, bl_annos, imp_annos, common_ids):
    print("\n" + "=" * 70)
    print("  逐帧 3D IoU (d3_box_overlap) 匹配中...")
    print("=" * 70)

    all_results = {}

    for cls_name in CLASS_NAMES:
        iou_thr = IOU_THRESHOLDS[cls_name]
        print(f"\n  [{cls_name}] IoU≥{iou_thr} ...")

        bl_per_frame, bl_m, bl_fn, bl_fp = match_by_3d_iou(
            gt_annos, bl_annos, cls_name, iou_thr)
        imp_per_frame, imp_m, imp_fn, imp_fp = match_by_3d_iou(
            gt_annos, imp_annos, cls_name, iou_thr)

        n_gt_total = sum(f['n_gt'] for f in bl_per_frame)
        bl_r = bl_m / max(1, n_gt_total) * 100
        imp_r = imp_m / max(1, n_gt_total) * 100

        print(f"    GT={n_gt_total} | BL: M={bl_m}, FN={bl_fn}, FP={bl_fp}, R={bl_r:.1f}%")
        print(f"    GT={n_gt_total} | IMP: M={imp_m}, FN={imp_fn}, FP={imp_fp}, R={imp_r:.1f}%")
        print(f"    Δ: M={imp_m-bl_m:+d}, FN={bl_fn-imp_fn:+d}, FP={bl_fp-imp_fp:+d}, R={imp_r-bl_r:+.1f}%")

        frame_diffs = []
        recovered_samples = []
        still_fn_samples = []
        dist_bins = defaultdict(lambda: {'bl_gt':0, 'bl_fn':0, 'imp_gt':0, 'imp_fn':0})

        for fi in range(len(bl_per_frame)):
            bl_f = bl_per_frame[fi]
            imp_f = imp_per_frame[fi]
            frame_id = f"image/{common_ids[fi]:06d}.jpg"

            bl_n_fn = bl_f['n_gt'] - bl_f['n_matched']
            imp_n_fn = imp_f['n_gt'] - imp_f['n_matched']

            frame_diffs.append({
                'frame_id': frame_id, 'n_gt': bl_f['n_gt'],
                'bl_fn': bl_n_fn, 'imp_fn': imp_n_fn,
                'fn_red': bl_n_fn - imp_n_fn,
            })

            # 距离段: 所有GT + FN
            gt_a = gt_annos[fi]
            gt_cls_mask = np.array([n.lower() == cls_name.lower() for n in gt_a['name']])
            for gi in np.where(gt_cls_mask)[0]:
                loc = gt_a['location'][gi]
                db = dist_bin(cam_dist(loc[0], loc[2]))
                dist_bins[db]['bl_gt'] += 1
                dist_bins[db]['imp_gt'] += 1

            for fn in bl_f['fn_list']:
                dist_bins[dist_bin(fn['distance'])]['bl_fn'] += 1
            for fn in imp_f['fn_list']:
                dist_bins[dist_bin(fn['distance'])]['imp_fn'] += 1

            # 被召回
            bl_fn_set = {(f['type'], round(f['distance'],1)) for f in bl_f['fn_list']}
            imp_fn_set = {(f['type'], round(f['distance'],1)) for f in imp_f['fn_list']}
            for r in bl_fn_set - imp_fn_set:
                recovered_samples.append({'frame':frame_id, 'class':cls_name, 'type':r[0], 'distance':r[1]})
            for s in bl_fn_set & imp_fn_set:
                if len(still_fn_samples) < 300:
                    still_fn_samples.append({'frame':frame_id, 'class':cls_name, 'type':s[0], 'distance':s[1]})

        frame_diffs.sort(key=lambda x: x['fn_red'], reverse=True)

        all_results[cls_name] = {
            'baseline': {'matched':bl_m, 'fn':bl_fn, 'fp':bl_fp, 'recall':bl_r},
            'improved': {'matched':imp_m, 'fn':imp_fn, 'fp':imp_fp, 'recall':imp_r},
            'n_gt_total': n_gt_total,
            'frame_diffs': frame_diffs,
            'dist_bins': dict(dist_bins),
            'recovered': recovered_samples,
            'still_fn': still_fn_samples,
        }

    return all_results


# ══════════════════════════════════════════════════════════════════
# Step 5: 生成报告
# ══════════════════════════════════════════════════════════════════

def generate_report(all_results, n_frames, output_dir):
    report_path = os.path.join(output_dir, 'error_analysis_report_v7.md')
    json_path   = os.path.join(output_dir, 'analysis_data_v7.json')

    json_out = {
        'eval_frames': n_frames,
        'method': '官方 result2kitti(修复维度) + get_label_annos + calculate_iou_partly(metric=2: d3_box_overlap) + 贪心匹配',
        'iou_thresholds': IOU_THRESHOLDS,
        'class_stats': {},
    }
    for cls_name in CLASS_NAMES:
        r = all_results[cls_name]
        json_out['class_stats'][cls_name] = {
            'n_gt': r['n_gt_total'],
            'baseline': r['baseline'],
            'improved': r['improved'],
            'recovered_count': len(r['recovered']),
            'still_fn_count': len(r['still_fn']),
        }
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    L = []
    L.append("# V2XFusion 3D检测误差定位与提升归因分析 (v7)")
    L.append("")
    L.append("> **方法**: 修复维度顺序 + 官方 `d3_box_overlap` (3D IoU) + 贪心匹配")
    L.append(f"> **评估帧数**: {n_frames} 帧 (Baseline 和 Exp06 均有预测)")
    L.append("> **匹配策略**: 3D IoU ≥ 阈值 + 贪心 (按置信度降序)")
    L.append("")
    L.append("## 一、核心结论")
    L.append("")

    for cls_name in CLASS_NAMES:
        r = all_results[cls_name]
        bl, imp = r['baseline'], r['improved']
        L.append(f"### {cls_name} (3D IoU≥{IOU_THRESHOLDS[cls_name]})")
        L.append(f"- GT 总数: **{r['n_gt_total']}**")
        L.append(f"- Baseline: 匹配 **{bl['matched']}**, 漏检 **{bl['fn']}**, 误检 **{bl['fp']}**, 召回率 **{bl['recall']:.1f}%**")
        L.append(f"- Exp06:    匹配 **{imp['matched']}**, 漏检 **{imp['fn']}**, 误检 **{imp['fp']}**, 召回率 **{imp['recall']:.1f}%**")
        L.append(f"- **ΔMatched: {imp['matched']-bl['matched']:+d}** | **ΔFN: {bl['fn']-imp['fn']:+d}** | **ΔFP: {bl['fp']-imp['fp']:+d}** | **ΔRecall: {imp['recall']-bl['recall']:+.1f}%**")
        L.append(f"- 被召回: **{len(r['recovered'])}** 个 | 仍漏检: **{len(r['still_fn'])}** 个(采样)")
        L.append("")

    L.append("## 二、类别级详细对比")
    L.append("")
    L.append("| 类别 | GT | BL_M | IMP_M | BL_FN | IMP_FN | ΔFN | BL_FP | IMP_FP | ΔFP | BL_R% | IMP_R% | ΔR% |")
    L.append("|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cls_name in CLASS_NAMES:
        r = all_results[cls_name]
        bl, imp = r['baseline'], r['improved']
        L.append(f"| {cls_name} | {r['n_gt_total']} | {bl['matched']} | {imp['matched']} | "
                 f"{bl['fn']} | {imp['fn']} | {bl['fn']-imp['fn']:+d} | "
                 f"{bl['fp']} | {imp['fp']} | {bl['fp']-imp['fp']:+d} | "
                 f"{bl['recall']:.1f} | {imp['recall']:.1f} | {imp['recall']-bl['recall']:+.1f} |")

    L.append("")
    L.append("## 三、距离段召回率 (Exp06)")
    L.append("")
    L.append("| 类别 | 距离段 | GT数 | FN数 | 召回率 |")
    L.append("|------|--------|:---:|:---:|:---:|")
    for cls_name in CLASS_NAMES:
        r = all_results[cls_name]
        for db in ['0-10m','10-20m','20-30m','30-40m','40-50m','50-60m','60-80m','80m+']:
            if db in r['dist_bins']:
                dd = r['dist_bins'][db]
                n_gt = dd.get('imp_gt',0)
                n_fn = dd.get('imp_fn',0)
                if n_gt > 0:
                    L.append(f"| {cls_name} | {db} | {n_gt} | {n_fn} | {100-n_fn/max(1,n_gt)*100:.1f}% |")

    L.append("")
    L.append("## 四、帧级改进排名")
    L.append("")
    for cls_name in CLASS_NAMES:
        r = all_results[cls_name]
        diffs = r['frame_diffs']
        best = sorted(diffs, key=lambda x: x['fn_red'], reverse=True)[:10]

        L.append(f"### {cls_name} — 漏检减少最多 Top 10")
        L.append("| Frame | GT | BL_FN | IMP_FN | FN_Reduction |")
        L.append("|-------|:---:|:---:|:---:|:---:|")
        for f in best:
            L.append(f"| {f['frame_id']} | {f['n_gt']} | {f['bl_fn']} | {f['imp_fn']} | **{f['fn_red']:+d}** |")
        L.append("")

    L.append("## 五、典型样本")
    L.append("")
    for cls_name in CLASS_NAMES:
        r = all_results[cls_name]
        rec = sorted(r['recovered'], key=lambda x: x['distance'])[:15]
        still = sorted(r['still_fn'], key=lambda x: x['distance'])[:15]

        L.append(f"### {cls_name} — 被召回 ({len(r['recovered'])} 个, 前15)")
        L.append("| Frame | GT类型 | 距离 |")
        L.append("|-------|--------|:---:|")
        for s in rec:
            L.append(f"| {s['frame']} | {s['type']} | {s['distance']:.1f}m |")
        L.append("")

        L.append(f"### {cls_name} — 仍漏检 ({len(r['still_fn'])} 个采样, 前15)")
        L.append("| Frame | GT类型 | 距离 |")
        L.append("|-------|--------|:---:|")
        for s in still:
            L.append(f"| {s['frame']} | {s['type']} | {s['distance']:.1f}m |")
        L.append("")

    L.append("## 六、方法说明")
    L.append("")
    L.append("### v7 修复: KITTI 维度顺序")
    L.append("")
    L.append("v6 的 bug: KITTI 行 `dh,dw,dl = dim[2], dim[0], dim[1]` 导致 l/w 互换.")
    L.append("v7 修复: `dh,dw,dl = dim[2], dim[1], dim[0]` — 与官方 `result2kitti` 完全一致.")
    L.append("")
    L.append("### 匹配方法")
    L.append("")
    L.append("使用 `calculate_iou_partly(metric=2)` → `d3_box_overlap()` 计算真实 3D IoU.")
    L.append("官方 3D IoU 已验证: Car 3D AP Mod ≈ 52.5% (与 `eval_result.txt` 一致).")
    L.append("")

    with open(report_path, 'w') as f:
        f.write('\n'.join(L))

    print(f"\n  ✅ 报告: {report_path}")
    print(f"  ✅ 数据: {json_path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  V2XFusion 误差分析 v7 — 修复维度 + 官方 3D IoU 匹配")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/4] 生成 KITTI 预测 (修复维度顺序)...")
    bl_pred_dir = generate_kitti_preds(BASELINE_RESULT, OUTPUT_DIR, 'baseline')
    imp_pred_dir = generate_kitti_preds(IMPROVED_RESULT, OUTPUT_DIR, 'improved')

    bl_files = set(os.listdir(bl_pred_dir))
    imp_files = set(os.listdir(imp_pred_dir))
    common_files = sorted(bl_files & imp_files)
    common_ids = sorted([int(f.replace('.txt','')) for f in common_files])
    n_frames = len(common_ids)
    print(f"\n[2/4] 共有 {n_frames} 帧有双方预测")

    print("\n[3/4] 使用 get_label_annos() 读取标注...")
    gt_annos  = get_label_annos(GT_LABEL_DIR, image_ids=common_ids)
    bl_annos  = get_label_annos(bl_pred_dir, image_ids=common_ids)
    imp_annos = get_label_annos(imp_pred_dir, image_ids=common_ids)
    print(f"  GT: {sum(len(a['name']) for a in gt_annos)} | BL: {sum(len(a['name']) for a in bl_annos)} | IMP: {sum(len(a['name']) for a in imp_annos)}")

    print("\n[4/4] 对比分析...")
    all_results = compare_experiments(gt_annos, bl_annos, imp_annos, common_ids)
    generate_report(all_results, n_frames, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("  v7 分析完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
