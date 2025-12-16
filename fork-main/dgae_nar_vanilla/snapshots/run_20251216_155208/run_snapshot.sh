#!/bin/bash
# 使用方法: bash run_snapshot.sh <你要运行的脚本> [参数...]
# 例如: bash run_snapshot.sh run_nar_1212.sh

TARGET_SCRIPT=$1
if [ -z "$TARGET_SCRIPT" ]; then
    echo "请指定要运行的脚本!"
    echo "用法: bash run_snapshot.sh run_nar_1212.sh"
    exit 1
fi

# 获取当前工作目录
WORK_DIR=$(pwd)
# 创建带时间戳的快照目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAP_DIR="${WORK_DIR}/snapshots/run_${TIMESTAMP}"

echo "=== 创建代码快照 ==="
echo "源目录: $WORK_DIR"
echo "快照目录: $SNAP_DIR"
mkdir -p "$SNAP_DIR"

# 1. 复制所有代码文件到快照目录
# 排除大型数据目录和输出目录，只复制代码
# -a: 归档模式 (保留权限、时间等)
# --exclude: 排除不需要复制的目录
rsync -a \
    --exclude 'models_own' \
    --exclude 'data' \
    --exclude 'wandb' \
    --exclude 'snapshots' \
    --exclude '.git' \
    --exclude '.cursor' \
    --exclude '__pycache__' \
    --exclude 'output_2026' \
    --exclude 'good_results' \
    . "$SNAP_DIR/"

# 2. 创建软链接指向原来的数据和输出目录
# 这样不仅节省空间，而且实验产生的 log/checkpoint 依然会保存在原来的 models_own 下
echo "=== 链接数据和输出目录 ==="

# 辅助函数：如果源目录存在，则创建软链接
link_dir() {
    local dir_name=$1
    if [ -d "${WORK_DIR}/${dir_name}" ]; then
        ln -s "${WORK_DIR}/${dir_name}" "$SNAP_DIR/${dir_name}"
        echo "Linked: ${dir_name}"
    fi
}

link_dir "models_own"
link_dir "data"
link_dir "wandb"
link_dir "output"
link_dir "plots"
link_dir "good_results"

# 3. 切换到快照目录并运行
echo "=== 启动任务 ==="
cd "$SNAP_DIR"
echo "正在快照目录中执行: $TARGET_SCRIPT"
echo "------------------------------------------------"

# 执行目标脚本，并传递所有参数
bash "$TARGET_SCRIPT" "${@:2}"

