#!/bin/sh

eval "$(conda shell.bash hook)"
conda activate monster-plus-plus
tensorboard --logdir ./logs/us3d/