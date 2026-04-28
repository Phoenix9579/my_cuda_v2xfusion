#!/bin/bash
# ================================================================
# 方案A快速验证脚本：用动态点云替换原始点云后评估
# 
# 使用方法：
#   bash scripts/eval_dynamic_pcd.sh /path/to/your/dynamic_pcd_directory
#
# 前提条件：
#   1. 你的动态PCD文件命名与原始 velodyne/ 下的文件一一对应
#      (如 000000.pcd, 000001.pcd, ...)
#   2. 坐标系保持一致（只删了背景点，没有旋转/平移）
#   3. conda v2xfusion 环境已激活
# ================================================================

set -e

DYNAMIC_PCD_DIR="$1"
ORIGINAL_VELODYNE="/mnt/BEVHeight-main/data/dair-v2x-i/velodyne"
BACKUP_VELODYNE="/mnt/BEVHeight-main/data/dair-v2x-i/velodyne_original"
CHECKPOINT="/mnt/bevfusion/checkpoints/dense_epoch_100_.pth"
CONFIG="configs/V2X-I/det/centerhead/lssfpn/camera+pointpillar/resnet34/default.yaml"
LOG_FILE="/mnt/bevfusion/answer/dynamic_pcd_eval_log.txt"

if [ -z "$DYNAMIC_PCD_DIR" ]; then
    echo "用法: bash scripts/eval_dynamic_pcd.sh /path/to/your/dynamic_pcd_directory"
    echo "请提供动态点云目录路径作为第一个参数"
    exit 1
fi

if [ ! -d "$DYNAMIC_PCD_DIR" ]; then
    echo "错误: 目录不存在 → $DYNAMIC_PCD_DIR"
    exit 1
fi

# 统计动态点云文件数
DYNAMIC_COUNT=$(ls "$DYNAMIC_PCD_DIR"/*.pcd 2>/dev/null | wc -l)
ORIGINAL_COUNT=$(ls "$ORIGINAL_VELODYNE"/*.pcd 2>/dev/null | wc -l)
echo "动态点云文件数: $DYNAMIC_COUNT"
echo "原始点云文件数: $ORIGINAL_COUNT"

if [ "$DYNAMIC_COUNT" -eq 0 ]; then
    echo "错误: 在 $DYNAMIC_PCD_DIR 中没有找到 .pcd 文件"
    exit 1
fi

# 步骤1：备份原始点云目录
if [ -L "$ORIGINAL_VELODYNE" ]; then
    echo "检测到 velodyne 已是软链接，先恢复原始目录..."
    rm "$ORIGINAL_VELODYNE"
    mv "$BACKUP_VELODYNE" "$ORIGINAL_VELODYNE"
fi

if [ ! -d "$BACKUP_VELODYNE" ]; then
    echo "备份原始 velodyne 目录..."
    mv "$ORIGINAL_VELODYNE" "$BACKUP_VELODYNE"
else
    echo "备份已存在，跳过备份步骤..."
    if [ -d "$ORIGINAL_VELODYNE" ] && [ ! -L "$ORIGINAL_VELODYNE" ]; then
        rm -rf "$ORIGINAL_VELODYNE"
    fi
fi

# 步骤2：创建软链接指向动态点云
echo "创建软链接: $ORIGINAL_VELODYNE → $DYNAMIC_PCD_DIR"
ln -s "$DYNAMIC_PCD_DIR" "$ORIGINAL_VELODYNE"

# 步骤3：运行评估
echo ""
echo "===== 开始动态点云评估 ====="
echo "日志输出到: $LOG_FILE"
cd /mnt/bevfusion

torchpack dist-run -np 1 python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --eval bbox 2>&1 | tee "$LOG_FILE"

# 步骤4：恢复原始目录
echo ""
echo "===== 恢复原始 velodyne 目录 ====="
rm "$ORIGINAL_VELODYNE"
mv "$BACKUP_VELODYNE" "$ORIGINAL_VELODYNE"
echo "原始目录已恢复"

echo ""
echo "===== 评估完成 ====="
echo "结果日志: $LOG_FILE"
echo "请对比 /mnt/bevfusion/answer/基线评估结果.md 中的数值"
