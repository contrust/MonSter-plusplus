#!/bin/sh

eval "$(conda shell.bash hook)"
conda activate monster-plus-plus
tensorboard --logdir ./logs/us3d/ --samples_per_plugin images=0
#tensorboard --logdir ./logs/us3d/