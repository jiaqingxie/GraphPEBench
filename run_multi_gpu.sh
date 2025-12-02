#!/bin/bash

# 多GPU并行训练脚本
# 在4个GPU上同时运行不同的seed或配置

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "用法: $0 <config_file>"
    echo "示例: $0 configs/GT/0_bench/GRIT/zinc/zinc-GRIT-noPE.yaml"
    exit 1
fi

echo "=== 多GPU并行训练 ==="
echo "配置文件: $CONFIG_FILE"
echo "使用4个GPU并行运行4个不同的seed"
echo ""

# 在4个GPU上分别运行4个不同seed的实验
CUDA_VISIBLE_DEVICES=0 python main.py --cfg $CONFIG_FILE --repeat 1 seed 0 > logs/gpu0.log 2>&1 &
PID0=$!
echo "GPU 0: PID=$PID0, seed=0"

CUDA_VISIBLE_DEVICES=1 python main.py --cfg $CONFIG_FILE --repeat 1 seed 1 > logs/gpu1.log 2>&1 &
PID1=$!
echo "GPU 1: PID=$PID1, seed=1"

CUDA_VISIBLE_DEVICES=2 python main.py --cfg $CONFIG_FILE --repeat 1 seed 2 > logs/gpu2.log 2>&1 &
PID2=$!
echo "GPU 2: PID=$PID2, seed=2"

CUDA_VISIBLE_DEVICES=3 python main.py --cfg $CONFIG_FILE --repeat 1 seed 3 > logs/gpu3.log 2>&1 &
PID3=$!
echo "GPU 3: PID=$PID3, seed=3"

echo ""
echo "所有任务已启动！"
echo "查看日志: tail -f logs/gpu*.log"
echo "停止所有任务: kill $PID0 $PID1 $PID2 $PID3"
echo ""

# 等待所有进程完成
wait $PID0 $PID1 $PID2 $PID3

echo "所有任务完成！"






