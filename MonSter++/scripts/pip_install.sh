#! /bin/bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.6.13
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install accelerate==1.0.1

# Configuration management and experiment tracking
pip install hydra-core
pip install omegaconf
pip install wandb

pip install gradio_imageslider
pip install gradio==4.29.0
pip install google-api-python-client
pip install google-auth-oauthlib
pip install google-auth-httplib2
pip install google-auth-oauthlib

# Dependencies for dataset visualization notebook
pip install numpy
pip install seaborn
pip install pillow
pip install rasterio
pip install pandas