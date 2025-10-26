#!/bin/sh

# Initialize conda without conda init
eval "$(conda shell.bash hook)"
conda activate monster
export PYTHONPATH=$(pwd)

SBATCH_CPU="-p hiperf --gres=gpu:4"
sbatch -n1 \
--cpus-per-task=1 \
--mem=32000 \
$SBATCH_CPU \
-t 1:00:00 \
--job-name=monster_jupyter_notebook_gpu_run \
--output=./logs/monster_jupyter_notebook_gpu_run_%j.log \
--error=./logs/monster_jupyter_notebook_gpu_run_%j.err \
\
--wrap="jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
