import glob
import os
import shutil
from torch.utils.data import random_split
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm
from PIL import Image
import random

SOURCE_IMAGES_ROOT = "/home/s0214/_scratch2/MonSter/datasets/us3d/images"
SOURCE_DISP_ROOT = "/home/s0214/_scratch2/MonSter/datasets/us3d/disp"
DEST_ROOT = "/home/s0214/_scratch2/MonSter-plusplus/MonSter++/datasets/us3d"
DEST_TRAIN_DIR = os.path.join(DEST_ROOT, "train")
DEST_VAL_DIR = os.path.join(DEST_ROOT, "val")
DEST_TEST_DIR = os.path.join(DEST_ROOT, "test")
SEED = 666
JAX_TRAIN_RATIO = 0.7
JAX_VAL_RATIO = 0.1
JAX_TEST_RATIO = 0.2
OMA_TRAIN_RATIO = 0.0
OMA_VAL_RATIO = 0.0
OMA_TEST_RATIO = 1.0
TRAIN_CROP_SIZE = (256, 1024)
SPLIT_INFO_BASENAME = "split_info.txt"

def get_cropped_array_list_by_size(arr: np.ndarray, size: Tuple[int, int]) -> Optional[List[np.ndarray]]:
    arr_h, arr_w = arr.shape[:2]
    crop_h, crop_w = size
    if arr_h < crop_h or arr_w < crop_w:
        return None
    cropped_arrs = []
    for i in range(0, arr_h, crop_h):
        for j in range(0, arr_w, crop_w):
            cropped_arrs.append(arr[i:i+crop_h, j:j+crop_w])
    return cropped_arrs

def add_suffix_before_extension(file_path: str, suffix: str) -> str:
    before_extension, extension = file_path.rsplit('.', 1)
    return f"{before_extension}{suffix}.{extension}"

def main():
    print("--------------------------------")
    print(f"Source images root: {SOURCE_IMAGES_ROOT}")
    print(f"Source disp root: {SOURCE_DISP_ROOT}")
    print(f"Dest train dir: {DEST_TRAIN_DIR}")
    print(f"Dest val dir: {DEST_VAL_DIR}")
    print(f"Dest test dir: {DEST_TEST_DIR}")
    print(f"Seed: {SEED}")
    print(f"JAX train ratio: {JAX_TRAIN_RATIO}")
    print(f"JAX val ratio: {JAX_VAL_RATIO}")
    print(f"JAX test ratio: {JAX_TEST_RATIO}")
    print(f"OMA train ratio: {OMA_TRAIN_RATIO}")
    print(f"OMA val ratio: {OMA_VAL_RATIO}")
    print(f"OMA test ratio: {OMA_TEST_RATIO}")
    print(f"Train crop size: {TRAIN_CROP_SIZE}")
    print("--------------------------------")
    print("Checking if source directories exist...")
    assert os.path.exists(SOURCE_IMAGES_ROOT)
    assert os.path.exists(SOURCE_DISP_ROOT)
    print("Done")
    print("Checking if destination directories exist...")
    print("Creating destination directories...")
    os.makedirs(DEST_TRAIN_DIR, exist_ok=True)
    os.makedirs(DEST_VAL_DIR, exist_ok=True)
    os.makedirs(DEST_TEST_DIR, exist_ok=True)
    print("Done")
    print("Reading image and disp lists...")
    image_left_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, '*LEFT*.tif')))
    image_right_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, '*RIGHT*.tif')))
    disp_list = sorted(glob.glob(os.path.join(SOURCE_DISP_ROOT, '*DSP*.tif')))
    assert len(image_left_list) == len(image_right_list) == len(disp_list)
    print(f"Found {len(image_left_list)} images")
    print("Splitting data into train, val, and test sets...")
    data_list = list(zip(image_left_list, image_right_list, disp_list))
    jax_train_list, jax_val_list, jax_test_list = random_split(data_list, [JAX_TRAIN_RATIO, JAX_VAL_RATIO, JAX_TEST_RATIO])
    oma_train_list, oma_val_list, oma_test_list = random_split(data_list, [OMA_TRAIN_RATIO, OMA_VAL_RATIO, OMA_TEST_RATIO])
    train_list = jax_train_list + oma_train_list
    val_list = jax_val_list + oma_val_list
    test_list = jax_test_list + oma_test_list
    print("Done")
    print("--------------------------------")
    print(f"JAX train split: {JAX_TRAIN_RATIO}, {len(jax_train_list)} JAX train images")
    print(f"JAX val split: {JAX_VAL_RATIO}, {len(jax_val_list)} JAX val images")
    print(f"JAX test split: {JAX_TEST_RATIO}, {len(jax_test_list)} JAX test images")
    print(f"OMA train split: {OMA_TRAIN_RATIO}, {len(oma_train_list)} OMA train images")
    print(f"OMA val split: {OMA_VAL_RATIO}, {len(oma_val_list)} OMA val images")
    print(f"OMA test split: {OMA_TEST_RATIO}, {len(oma_test_list)} OMA test images")
    print("--------------------------------")
    print("Cropping the train images and saving to the train directory...")
    for (image_left_path, image_right_path, disp_path) in tqdm(train_list, desc="train", unit="image"):
        image_left = Image.open(image_left_path)
        image_right = Image.open(image_right_path)
        disp = Image.open(disp_path)
        cropped_left_images = get_cropped_array_list_by_size(np.array(image_left), TRAIN_CROP_SIZE)
        cropped_right_images = get_cropped_array_list_by_size(np.array(image_right), TRAIN_CROP_SIZE)
        cropped_disp_images = get_cropped_array_list_by_size(np.array(disp, dtype=np.float32), TRAIN_CROP_SIZE)
        if cropped_left_images is not None and cropped_right_images is not None and cropped_disp_images is not None:
            for i, (cropped_left_image, cropped_right_image, cropped_disp_image) in enumerate(zip(cropped_left_images, cropped_right_images, cropped_disp_images)):
                left_image_path = os.path.join(DEST_TRAIN_DIR, add_suffix_before_extension(os.path.basename(image_left_path), f"_{i}"))
                right_image_path = os.path.join(DEST_TRAIN_DIR, add_suffix_before_extension(os.path.basename(image_right_path), f"_{i}"))
                disp_image_path = os.path.join(DEST_TRAIN_DIR, add_suffix_before_extension(os.path.basename(disp_path), f"_{i}"))
                Image.fromarray(cropped_left_image).convert("RGB").save(left_image_path)
                Image.fromarray(cropped_right_image).convert("RGB").save(right_image_path)
                Image.fromarray(cropped_disp_image).convert("F").save(disp_image_path)
    print("Done")
    print("Saving the val images to the val directory...")
    for (image_left_path, image_right_path, disp_path) in tqdm(val_list, desc="val", unit="image"):
        shutil.copy(image_left_path, os.path.join(DEST_VAL_DIR, os.path.basename(image_left_path)))
        shutil.copy(image_right_path, os.path.join(DEST_VAL_DIR, os.path.basename(image_right_path)))
        shutil.copy(disp_path, os.path.join(DEST_VAL_DIR, os.path.basename(disp_path)))
    print("Done")
    print("Saving the test images to the test directory...")
    for (image_left_path, image_right_path, disp_path) in tqdm(test_list, desc="test", unit="image"):
        shutil.copy(image_left_path, os.path.join(DEST_TEST_DIR, os.path.basename(image_left_path)))
        shutil.copy(image_right_path, os.path.join(DEST_TEST_DIR, os.path.basename(image_right_path)))
        shutil.copy(disp_path, os.path.join(DEST_TEST_DIR, os.path.basename(disp_path)))
    print("Done")
    print("Saving the split information to the split info file...")
    with open(os.path.join(DEST_ROOT, SPLIT_INFO_BASENAME), "w") as f:
        f.write(f"JAX train ratio: {JAX_TRAIN_RATIO}\n")
        f.write(f"JAX val ratio: {JAX_VAL_RATIO}\n")
        f.write(f"JAX test ratio: {JAX_TEST_RATIO}\n")
        f.write(f"OMA train ratio: {OMA_TRAIN_RATIO}\n")
        f.write(f"OMA val ratio: {OMA_VAL_RATIO}\n")
        f.write(f"OMA test ratio: {OMA_TEST_RATIO}\n")
        f.write(f"Train list length: {len(train_list)}\n")
        f.write(f"Val list length: {len(val_list)}\n")
        f.write(f"Test list length: {len(test_list)}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Train crop size: {TRAIN_CROP_SIZE}\n")
    print("Done")
    print("--------------------------------")
    print("The US3D dataset has been split successfully")
    print("--------------------------------")

if __name__ == "__main__":
    main()