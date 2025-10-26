#!/bin/sh

SBATCH_CPU=""
SBATCH_CPU="-p hiperf --gres=gpu:8"

sbatch -n1 \
--cpus-per-task=10 \
--mem=100000 \
$SBATCH_CPU \
-t 20:00:00 \
--job-name=monster_us3d_train \
--output=./logs/monster_datasets_us3d_train_%j.log \
--error=./logs/monster_datasets_us3d_train_%j.err \
\
--wrap="PATH=/home/s0214/.conda/envs/monster-plus-plus/bin:\$PATH CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --num_machines=1 --mixed_precision=no --dynamo_backend=no train_us3d.py"