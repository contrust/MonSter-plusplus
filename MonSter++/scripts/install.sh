#! /bin/bash
conda create -n monster-plus-plus python=3.9 -y
conda activate monster-plus-plus
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install --no-cache-dir tqdm scipy opencv-python scikit-image tensorboard matplotlib timm==0.6.13 accelerate==1.0.1 gradio_imageslider gradio==4.29.0 openexr pyexr imath h5py omegaconf hydra-core     