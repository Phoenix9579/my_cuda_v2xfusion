"""
深度分析检测结果：按原始类别、距离分段、目标大小分别统计AP/召回率
用法：python scripts/analyze_detections.py outputs/results_nusc.json
"""
import json
import sys
import os
import numpy as np

def analyze_nusc_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 统计各原始类别的检测数量和分数分布
    class_stats = {}
    distance_buckets = {'0-20m': 0, '20-40m': 0, '40-60m': 0, '60-80m': 0, '80m+': 0}
    class_distance = {}
    
    for sample_token, preds in results.items():
        for pred in preds:
            cls = pred['detection_name']
            score = pred['detection_score']
            x, y = pred['translation'][0], pred['translation'][1]
            dist = np.sqrt(x**2 + y**2)
            
            if cls not in class_stats:
                class_stats[cls] = {'count': 0, 'scores': [], 'distances': []}
                class_distance[cls] = {k: 0 for k in distance_buckets}
            
            class_stats[cls]['count'] += 1
            class_stats[cls]['scores'].append(score)
            class_stats[cls]['distances'].append(dist)
            
            # 距离分段
            if dist < 20:    class_distance[cls]['0-20m'] += 1
            elif dist < 40:  class_distance[cls]['20-40m'] += 1
            elif dist < 60:  class_distance[cls]['40-60m'] += 1
            elif dist < 80:  class_distance[cls]['60-80m'] += 1
            else:            class_distance[cls]['80m+'] += 1
    
    print("=" * 70)
    print(f"分析文件: {os.path.basename(json_path)}")
    print(f"总样本数: {len(results)}")
    print("=" * 70)
    
    print("\n【各原始类别检测统计】")
    print(f"{'类别':<25} {'检测数':>8} {'均值分':>8} {'中位分':>8} {'>0.45数':>8}")
    print("-" * 62)
    
    total = 0
    for cls in sorted(class_stats.keys(), key=lambda x: -class_stats[x]['count']):
        st = class_stats[cls]
        scores = np.array(st['scores'])
        high_score = (scores > 0.45).sum()
        total += st['count']
        print(f"{cls:<25} {st['count']:>8} {scores.mean():>8.3f} {np.median(scores):>8.3f} {high_score:>8}")
    print(f"{'合计':<25} {total:>8}")
    
    print("\n【各类别距离分布（>0.45分的检测）】")
    header = f"{'类别':<20}" + "".join(f"{k:>10}" for k in distance_buckets)
    print(header)
    print("-" * 70)
    
    for cls in ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle']:
        if cls not in class_distance:
            continue
        st = class_stats[cls]
        scores = np.array(st['scores'])
        distances = np.array(st['distances'])
        # 只统计高分检测
        mask = scores > 0.45
        valid_dists = distances[mask]
        
        dist_cnt = {k: 0 for k in distance_buckets}
        for d in valid_dists:
            if d < 20:    dist_cnt['0-20m'] += 1
            elif d < 40:  dist_cnt['20-40m'] += 1
            elif d < 60:  dist_cnt['40-60m'] += 1
            elif d < 80:  dist_cnt['60-80m'] += 1
            else:         dist_cnt['80m+'] += 1
        
        row = f"{cls:<20}" + "".join(f"{dist_cnt[k]:>10}" for k in distance_buckets)
        print(row)
    
    print("\n【KITTI类别映射说明】")
    print("  Car    ← car, van, truck, bus (4类合并)")
    print("  Cyclist ← bicycle, motorcycle, trailer (3类合并)")
    print("  Pedestrian ← pedestrian")
    print("  ❌ 丢弃: barrier, traffic_cone, construction_vehicle")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/bevfusion/outputs/results_nusc.json'
    analyze_nusc_results(path)
