#!/bin/bash

#SBATCH --job-name=vqgpt2_all_layers
#SBATCH --partition=A100,ADA6000,L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=luckywang@sjtu.edu.cn

export NCCL_P2P_DISABLE=1
RUN_NAME=$1
OUTPUT_DIR=./exp/$RUN_NAME

torchrun --nproc_per_node=4 train_vqvaegpt2_all_layers.py \
--vq_dir $OUTPUT_DIR \
2>&1 | tee $OUTPUT_DIR/trainvqgpt2.log

torchrun --nproc_per_node=4 train_vqvaegpt2_all_layers.py \
--vq_dir $OUTPUT_DIR \
--test \
2>&1 | tee -a $OUTPUT_DIR/testvqgpt2.log