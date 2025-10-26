#!/bin/bash

# Depth-Anything-V2 Pretrained Weights Download Script
# This script downloads the pretrained weights for Depth-Anything-V2 models

echo "=== Depth-Anything-V2 Pretrained Weights Download Script ==="
echo "This script will download the pretrained weights for Depth-Anything-V2 models."

# Create pretrained directory if it doesn't exist
mkdir -p pretrained

# Download Depth-Anything-V2 weights
echo "Downloading Depth-Anything-V2 weights..."

# Download from the official Depth-Anything-V2 repository
# These are the weights from the official Depth-Anything-V2 repository
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true -O pretrained/depth_anything_v2_vitl.pth

echo "=== Download Complete ==="
echo "Depth-Anything-V2 pretrained weights have been downloaded successfully!"
echo "Files downloaded:"
ls -la pretrained/depth_anything_v2_*.pth 