#!/bin/bash

# 在4个GPU上并行运行4个不同的实验
# 每个GPU运行一个独立的配置或seed

# 创建日志目录
mkdir -p logs

# 定义4个配置文件或使用同一个配置但不同seed
CONFIG1=${1:-"configs/GT/MLP/LRGB/peptides_func/peptides-func-MLP-noPE.yaml"}
CONFIG2=${2:-$CONFIG1}
CONFIG3=${3:-$CONFIG1}
CONFIG4=${4:-$CONFIG1}

echo "=== 4-GPU并行训练 ==="
echo "GPU 0: $CONFIG1"
echo "GPU 1: $CONFIG2"
echo "GPU 2: $CONFIG3"
echo "GPU 3: $CONFIG4"
echo "================================"

# GPU 0
CUDA_VISIBLE_DEVICES=0 python main.py --cfg $CONFIG1 accelerator cuda:0 > logs/gpu0_$(basename $CONFIG1 .yaml).log 2>&1 &
PID0=$!
echo "[GPU 0] PID=$PID0 started"

# GPU 1
CUDA_VISIBLE_DEVICES=1 python main.py --cfg $CONFIG2 accelerator cuda:0 > logs/gpu1_$(basename $CONFIG2 .yaml).log 2>&1 &
PID1=$!
echo "[GPU 1] PID=$PID1 started"

# GPU 2
CUDA_VISIBLE_DEVICES=2 python main.py --cfg $CONFIG3 accelerator cuda:0 > logs/gpu2_$(basename $CONFIG3 .yaml).log 2>&1 &
PID2=$!
echo "[GPU 2] PID=$PID2 started"

# GPU 3
CUDA_VISIBLE_DEVICES=3 python main.py --cfg $CONFIG4 accelerator cuda:0 > logs/gpu3_$(basename $CONFIG4 .yaml).log 2>&1 &
PID3=$!
echo "[GPU 3] PID=$PID3 started"

echo ""
echo "所有任务已启动！"
echo "================================"
echo "查看实时日志："
echo "  GPU 0: tail -f logs/gpu0_*.log"
echo "  GPU 1: tail -f logs/gpu1_*.log"
echo "  GPU 2: tail -f logs/gpu2_*.log"
echo "  GPU 3: tail -f logs/gpu3_*.log"
echo ""
echo "查看所有日志: tail -f logs/*.log"
echo ""
echo "停止所有任务:"
echo "  kill $PID0 $PID1 $PID2 $PID3"
echo "================================"

# 保存PID到文件
echo "$PID0 $PID1 $PID2 $PID3" > logs/multi_gpu_pids.txt

# 等待所有进程完成
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "所有任务完成！"






