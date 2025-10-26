#!/bin/sh

# Initialize conda without conda init
eval "$(conda shell.bash hook)"
conda activate monster
export PYTHONPATH=$(pwd)

CPU_TRAIN=false

SBATCH_CPU=""
PYTHON_CPU=""
if [ $CPU_TRAIN == false ]
then
  SBATCH_CPU="-p hiperf --gres=gpu:1"
else
  PYTHON_CPU="--cpu"
fi

# Default checkpoint path - modify as needed
CHECKPOINT_PATH="./checkpoints/mix_all.pth"
DATASET="whu"

sbatch -n1 \
--cpus-per-task=10 \
--mem=32000 \
$SBATCH_CPU \
-t 1:00:00 \
--job-name=datasets_whu_evaluate \
--output=./logs/datasets_whu_evaluate_%j.log \
--error=./logs/datasets_whu_evaluate_%j.err \
\
--wrap="python evaluate_stereo.py --dataset $DATASET --restore_ckpt $CHECKPOINT_PATH"