#!/bin/bash

# KITTI Datasets Download Script
# This script downloads and organizes KITTI 2012 and 2015 datasets for stereo evaluation

set -e  # Exit on any error

echo "=== KITTI Datasets Download Script ==="
echo "This script will download KITTI 2012 and 2015 datasets and organize them properly."
echo ""

# Create base directories
echo "Creating directory structure..."
mkdir -p datasets/kitti/2012
mkdir -p datasets/kitti/2015

# Function to download and extract dataset
download_and_extract() {
    local url=$1
    local output_dir=$2
    local filename=$(basename $url)
    local filepath="$output_dir/$filename"
    
    echo "Downloading $filename..."
    if [ -f "$filepath" ]; then
        echo "File already exists, skipping download."
    else
        wget -c "$url" -O "$filepath"
    fi
    
    echo "Extracting $filename..."
    unzip -q "$filepath" -d "$output_dir"
    
    echo "Cleaning up zip file..."
    rm "$filepath"
}

# Download KITTI 2012 dataset
echo ""
echo "=== Downloading KITTI 2012 Dataset ==="
KITTI_2012_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"
download_and_extract "$KITTI_2012_URL" "datasets/kitti/2012"

# Download KITTI 2015 dataset  
echo ""
echo "=== Downloading KITTI 2015 Dataset ==="
KITTI_2015_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_multiview.zip"
download_and_extract "$KITTI_2015_URL" "datasets/kitti/2015"

# Organize KITTI 2012 data
echo ""
echo "=== Organizing KITTI 2012 data ==="
if [ -d "datasets/kitti/2012/data_scene_flow" ]; then
    echo "Moving KITTI 2012 files to correct structure..."
    # Move training data
    if [ -d "datasets/kitti/2012/data_scene_flow/training" ]; then
        mv datasets/kitti/2012/data_scene_flow/training/* datasets/kitti/2012/
        rmdir datasets/kitti/2012/data_scene_flow/training
    fi
    
    # Move testing data
    if [ -d "datasets/kitti/2012/data_scene_flow/testing" ]; then
        mv datasets/kitti/2012/data_scene_flow/testing/* datasets/kitti/2012/
        rmdir datasets/kitti/2012/data_scene_flow/testing
    fi
    
    # Remove empty directory
    rmdir datasets/kitti/2012/data_scene_flow
fi

# Organize KITTI 2015 data
echo ""
echo "=== Organizing KITTI 2015 data ==="
if [ -d "datasets/kitti/2015/data_scene_flow_2015" ]; then
    echo "Moving KITTI 2015 files to correct structure..."
    # Move training data
    if [ -d "datasets/kitti/2015/data_scene_flow_2015/training" ]; then
        mv datasets/kitti/2015/data_scene_flow_2015/training/* datasets/kitti/2015/
        rmdir datasets/kitti/2015/data_scene_flow_2015/training
    fi
    
    # Move testing data
    if [ -d "datasets/kitti/2015/data_scene_flow_2015/testing" ]; then
        mv datasets/kitti/2015/data_scene_flow_2015/testing/* datasets/kitti/2015/
        rmdir datasets/kitti/2015/data_scene_flow_2015/testing
    fi
    
    # Remove empty directory
    rmdir datasets/kitti/2015/data_scene_flow_2015
fi

# Verify the structure
echo ""
echo "=== Verifying directory structure ==="
echo "Checking KITTI 2012 structure..."
if [ -d "datasets/kitti/2012/training/colored_0" ] && [ -d "datasets/kitti/2012/training/colored_1" ] && [ -d "datasets/kitti/2012/training/disp_occ" ]; then
    echo "✓ KITTI 2012 training structure is correct"
    echo "  - Left images: $(ls datasets/kitti/2012/training/colored_0/*.png | wc -l) files"
    echo "  - Right images: $(ls datasets/kitti/2012/training/colored_1/*.png | wc -l) files"
    echo "  - Disparity maps: $(ls datasets/kitti/2012/training/disp_occ/*.png | wc -l) files"
else
    echo "✗ KITTI 2012 training structure is incorrect"
fi

if [ -d "datasets/kitti/2012/testing/colored_0" ] && [ -d "datasets/kitti/2012/testing/colored_1" ]; then
    echo "✓ KITTI 2012 testing structure is correct"
    echo "  - Left images: $(ls datasets/kitti/2012/testing/colored_0/*.png | wc -l) files"
    echo "  - Right images: $(ls datasets/kitti/2012/testing/colored_1/*.png | wc -l) files"
else
    echo "✗ KITTI 2012 testing structure is incorrect"
fi

echo ""
echo "Checking KITTI 2015 structure..."
if [ -d "datasets/kitti/2015/training/image_2" ] && [ -d "datasets/kitti/2015/training/image_3" ] && [ -d "datasets/kitti/2015/training/disp_occ_0" ]; then
    echo "✓ KITTI 2015 training structure is correct"
    echo "  - Left images: $(ls datasets/kitti/2015/training/image_2/*.png | wc -l) files"
    echo "  - Right images: $(ls datasets/kitti/2015/training/image_3/*.png | wc -l) files"
    echo "  - Disparity maps: $(ls datasets/kitti/2015/training/disp_occ_0/*.png | wc -l) files"
else
    echo "✗ KITTI 2015 training structure is incorrect"
fi

if [ -d "datasets/kitti/2015/testing/image_2" ] && [ -d "datasets/kitti/2015/testing/image_3" ]; then
    echo "✓ KITTI 2015 testing structure is correct"
    echo "  - Left images: $(ls datasets/kitti/2015/testing/image_2/*.png | wc -l) files"
    echo "  - Right images: $(ls datasets/kitti/2015/testing/image_3/*.png | wc -l) files"
else
    echo "✗ KITTI 2015 testing structure is incorrect"
fi

echo ""
echo "=== Download Complete ==="
echo "KITTI datasets have been downloaded and organized successfully!"
echo "You can now use them with evaluate_stereo.py"
echo ""
echo "Expected directory structure:"
echo "datasets/kitti/"
echo "├── 2012/"
echo "│   ├── training/"
echo "│   │   ├── colored_0/     (left images)"
echo "│   │   ├── colored_1/     (right images)"
echo "│   │   └── disp_occ/      (disparity maps)"
echo "│   └── testing/"
echo "│       ├── colored_0/     (left images)"
echo "│       └── colored_1/     (right images)"
echo "└── 2015/"
echo "    ├── training/"
echo "    │   ├── image_2/       (left images)"
echo "    │   ├── image_3/       (right images)"
echo "    │   └── disp_occ_0/    (disparity maps)"
echo "    └── testing/"
echo "        ├── image_2/       (left images)"
echo "        └── image_3/       (right images)" 