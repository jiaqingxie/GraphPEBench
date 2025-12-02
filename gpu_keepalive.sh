#!/bin/bash

# GPU Keep-Alive Shell Script
# 防止GPU因利用率过低自动关机

echo "=== GPU保活脚本启动 ==="

# 默认参数
GPU_ID=1
UTILIZATION=60
INTERVAL=60

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --util)
            UTILIZATION="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --gpu ID        指定GPU设备ID (默认: 1)"
            echo "  --util PERCENT  目标利用率百分比 (默认: 60)"
            echo "  --interval SEC  检查间隔秒数 (默认: 60)"
            echo "  --help|-h       显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                    # 使用默认参数"
            echo "  $0 --gpu 1 --util 50 # 使用GPU 1，50%利用率"
            echo "  $0 --interval 30     # 30秒检查间隔"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "参数设置:"
echo "  GPU设备: $GPU_ID"
echo "  目标利用率: $UTILIZATION%"
echo "  检查间隔: ${INTERVAL}秒"
echo ""

# 检查Python脚本是否存在
if [ ! -f "gpu_keepalive.py" ]; then
    echo "错误: gpu_keepalive.py 文件不存在"
    exit 1
fi

# 运行Python脚本
echo "启动GPU保活..."
echo "按 Ctrl+C 停止"
echo "=========================="

python3 gpu_keepalive.py --gpu $GPU_ID --util $UTILIZATION --interval $INTERVAL
