#!/bin/bash


cd /root/PEGT 

#!/usr/bin/env bash
set -euo pipefail

# 设置CPU线程数（替代SLURM的--cpus-per-task=4）
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 初始化 conda（通用写法）
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENVIRONMENT=/tos-bjml-ai4chem/xiejiaqing/graph
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"

# Execute your code

#python main.py --cfg configs/GT/2_MPNN/GIN/mnist/mnist-GIN-COREGD.yaml  wandb.use True accelerator "cuda:0" seed 7

#python main.py --cfg configs/GT/2_MPNN/GIN/mnist/mnist-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN/cifar10/cifar10-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7

#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 0
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-noPE.yaml  wandb.use True accelerator "cuda:0" seed 0
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-noPE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-LapPE.yaml  wandb.use True accelerator "cuda:0" seed 0
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-LapPE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-RWSE.yaml  wandb.use True accelerator "cuda:0" seed 0
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-RWSE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-GCKN.yaml  wandb.use True accelerator "cuda:0" seed 0
#python main.py --cfg configs/GT/2_MPNN/GIN_full/synthetic/fourcycles/fourcycles-GIN-GCKN.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN/synthetic/skipcircles/skipcircles-GIN-RRWP.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GatedGCN/zinc/zinc-GatedGCN-RRWP.yaml  wandb.use True accelerator "cuda:0" seed 0
#
python main.py --cfg configs/GT/MLP/LRGB/COCO/coco-MLP-noPE.yaml wandb.use False accelerator "cuda:1" seed 0

#python main.py --cfg configs/GT/2_MPNN/GIN/cifar10/cifar10-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN/cluster/cluster-GIN-LapPE.yaml  wandb.use True accelerator "cuda:0" seed 100
#python main.py --cfg configs/GT/2_MPNN/GIN/zinc/zinc-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 42
#python main.py --cfg configs/GT/2_MPNN/GIN/mnist/mnist-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN/pattern/pattern-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7

#python main.py --cfg configs/GT/2_MPNN/GIN/zinc/zinc-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 0
#python main.py --cfg configs/GT/2_MPNN/GIN/mnist/mnist-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7

#python main.py --cfg configs/GT/2_MPNN/GIN/cifar10/cifar10-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 0

#python main.py --cfg configs/GT/2_MPNN/GIN/LRGB/peptides_struct/peptides-struct-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 0
#python main.py --cfg configs/GT/2_MPNN/GIN/LRGB/VOC/voc-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN/LRGB/pcqm_contact/pcqm-contact-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7



#python main.py --cfg configs/GT/2_MPNN/GIN/cifar10/cifar10-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 100
#python main.py --cfg configs/GT/2_MPNN/GIN/LRGB/peptides_struct/peptides-struct-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN/LRGB/VOC/voc-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 7
#python main.py --cfg configs/GT/2_MPNN/GIN/LRGB/VOC/voc-GIN-GPSE.yaml  wandb.use True accelerator "cuda:0" seed 0


echo "Finished at: $(date)"

# End the script with exit code 0
exit 0