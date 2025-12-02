#!/bin/bash

# 使用DataParallel在多GPU上训练
# 注意：这需要设置环境变量

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "用法: $0 <config_file>"
    echo "示例: $0 configs/GT/0_bench/GRIT/zinc/zinc-GRIT-noPE.yaml"
    exit 1
fi

# 设置可用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== DataParallel多GPU训练 ==="
echo "配置文件: $CONFIG_FILE"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# 运行训练
python main.py --cfg $CONFIG_FILE accelerator cuda:0

echo "训练完成！"






