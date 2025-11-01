import glob
import os
import shutil
import torch
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
JAX_TILE_TRAIN_RATIO = 0.7
JAX_TILE_VAL_RATIO = 0.1
JAX_TILE_TEST_RATIO = 0.2
OMA_TILE_TRAIN_RATIO = 0.00
OMA_TILE_VAL_RATIO = 0.00
OMA_TILE_TEST_RATIO = 1.00
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

def get_tile_number_from_file_path(file_path: str) -> str:
    basename = os.path.basename(file_path)
    return basename.split("_")[1]

def get_unique_tile_numbers_from_file_paths(file_paths: List[str]) -> List[str]:
    tile_numbers_set = set()
    for file_path in file_paths:
        tile_number = get_tile_number_from_file_path(file_path)
        tile_numbers_set.add(tile_number)
    tile_numbers = list(tile_numbers_set)
    tile_numbers.sort()
    return tile_numbers

def get_data_split_by_tile_numbers(data_list: List[Tuple[str, str, str]],
                                   train_tile_numbers: List[str],
                                   val_tile_numbers: List[str],
                                   test_tile_numbers: List[str]) -> List[Tuple[str, str, str]]:
    train_data_list = []
    val_data_list = []
    test_data_list = []
    train_tile_numbers_set = set(train_tile_numbers)
    val_tile_numbers_set = set(val_tile_numbers)
    test_tile_numbers_set = set(test_tile_numbers)
    for data in data_list:
        tile_number = get_tile_number_from_file_path(data[0])
        if tile_number in train_tile_numbers_set:
            train_data_list.append(data)
        elif tile_number in val_tile_numbers_set:
            val_data_list.append(data)
        elif tile_number in test_tile_numbers_set:
            test_data_list.append(data)
        else:
            print(f"Tile number {tile_number} not found in train, val, or test sets")
    return train_data_list, val_data_list, test_data_list

def main():
    print("--------------------------------")
    print(f"Source images root: {SOURCE_IMAGES_ROOT}")
    print(f"Source disp root: {SOURCE_DISP_ROOT}")
    print(f"Dest train dir: {DEST_TRAIN_DIR}")
    print(f"Dest val dir: {DEST_VAL_DIR}")
    print(f"Dest test dir: {DEST_TEST_DIR}")
    print(f"Seed: {SEED}")
    print(f"JAX tile train ratio: {JAX_TILE_TRAIN_RATIO}")
    print(f"JAX tile val ratio: {JAX_TILE_VAL_RATIO}")
    print(f"JAX tile test ratio: {JAX_TILE_TEST_RATIO}")
    print(f"OMA tile train ratio: {OMA_TILE_TRAIN_RATIO}")
    print(f"OMA tile val ratio: {OMA_TILE_VAL_RATIO}")
    print(f"OMA tile test ratio: {OMA_TILE_TEST_RATIO}")
    print(f"Train crop size: {TRAIN_CROP_SIZE}")
    print("--------------------------------")
    print("Checking if source directories exist...")
    assert os.path.exists(SOURCE_IMAGES_ROOT)
    assert os.path.exists(SOURCE_DISP_ROOT)
    print("Done")
    print("Setting seed...")
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    print("Done")
    print("Checking if source directories exist...")
    assert os.path.exists(SOURCE_IMAGES_ROOT)
    assert os.path.exists(SOURCE_DISP_ROOT)
    print("Done")
    print("Checking if destination directories exist...")
    if os.path.exists(DEST_TRAIN_DIR):
        answer = input("Dest train dir already exists, do you want to continue? (y/n) ")
        if str.lower(answer) != "y":
            print("Exiting...")
            exit(0)
        shutil.rmtree(DEST_TRAIN_DIR)
    if os.path.exists(DEST_VAL_DIR):
        answer = input("Dest val dir already exists, do you want to continue? (y/n) ")
        if str.lower(answer) != "y":
            print("Exiting...")
            exit(0)
        shutil.rmtree(DEST_VAL_DIR)
    if os.path.exists(DEST_TEST_DIR):
        answer = input("Dest test dir already exists, do you want to continue? (y/n) ")
        if str.lower(answer) != "y":
            print("Exiting...")
            exit(0)
        shutil.rmtree(DEST_TEST_DIR)
    print("Creating destination directories...")
    os.makedirs(DEST_TRAIN_DIR, exist_ok=True)
    os.makedirs(DEST_VAL_DIR, exist_ok=True)
    os.makedirs(DEST_TEST_DIR, exist_ok=True)
    print("Done")
    print("Reading image and disp lists...")
    jax_image_left_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, 'JAX*LEFT*.tif')))
    jax_image_right_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, 'JAX*RIGHT*.tif')))
    jax_disp_list = sorted(glob.glob(os.path.join(SOURCE_DISP_ROOT, 'JAX*DSP*.tif')))
    oma_image_left_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, 'OMA*LEFT*.tif')))
    oma_image_right_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, 'OMA*RIGHT*.tif')))
    oma_disp_list = sorted(glob.glob(os.path.join(SOURCE_DISP_ROOT, 'OMA*DSP*.tif')))
    assert len(jax_image_left_list) == len(jax_image_right_list) == len(jax_disp_list)
    assert len(oma_image_left_list) == len(oma_image_right_list) == len(oma_disp_list)
    print(f"Found {len(jax_image_left_list)} JAX images")
    print(f"Found {len(oma_image_left_list)} OMA images")
    jax_data_list = list(zip(jax_image_left_list, jax_image_right_list, jax_disp_list))
    oma_data_list = list(zip(oma_image_left_list, oma_image_right_list, oma_disp_list))
    jax_tile_numbers = get_unique_tile_numbers_from_file_paths(jax_image_left_list)
    oma_tile_numbers = get_unique_tile_numbers_from_file_paths(oma_image_left_list)
    print(f"Found {len(jax_tile_numbers)} JAX tile numbers")
    print(f"Found {len(oma_tile_numbers)} OMA tile numbers")
    print("Splitting data into train, val, and test sets...")
    jax_train_tile_numbers, jax_val_tile_numbers, jax_test_tile_numbers = random_split(jax_tile_numbers, [JAX_TILE_TRAIN_RATIO, JAX_TILE_VAL_RATIO, JAX_TILE_TEST_RATIO])
    oma_train_tile_numbers, oma_val_tile_numbers, oma_test_tile_numbers = random_split(oma_tile_numbers, [OMA_TILE_TRAIN_RATIO, OMA_TILE_VAL_RATIO, OMA_TILE_TEST_RATIO])
    jax_train_data_list, jax_val_data_list, jax_test_data_list = get_data_split_by_tile_numbers(jax_data_list, jax_train_tile_numbers, jax_val_tile_numbers, jax_test_tile_numbers)
    oma_train_data_list, oma_val_data_list, oma_test_data_list = get_data_split_by_tile_numbers(oma_data_list, oma_train_tile_numbers, oma_val_tile_numbers, oma_test_tile_numbers)
    train_data_list = jax_train_data_list + oma_train_data_list
    val_data_list = jax_val_data_list + oma_val_data_list
    test_data_list = jax_test_data_list + oma_test_data_list
    print("Done")
    print("--------------------------------")
    print(f"JAX tile train split: {JAX_TILE_TRAIN_RATIO}, {len(jax_train_data_list)} images")
    print(f"JAX tile val split: {JAX_TILE_VAL_RATIO}, {len(jax_val_data_list)} images")
    print(f"JAX tile test split: {JAX_TILE_TEST_RATIO}, {len(jax_test_data_list)} images")
    print(f"OMA tile train split: {OMA_TILE_TRAIN_RATIO}, {len(oma_train_data_list)} images")
    print(f"OMA tile val split: {OMA_TILE_VAL_RATIO}, {len(oma_val_data_list)} images")
    print(f"OMA tile test split: {OMA_TILE_TEST_RATIO}, {len(oma_test_data_list)} images")
    print("--------------------------------")
    print("Cropping the train images and saving to the train directory...")
    for (image_left_path, image_right_path, disp_path) in tqdm(train_data_list, desc="train", unit="image"):
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
    for (image_left_path, image_right_path, disp_path) in tqdm(val_data_list, desc="val", unit="image"):
        shutil.copy(image_left_path, os.path.join(DEST_VAL_DIR, os.path.basename(image_left_path)))
        shutil.copy(image_right_path, os.path.join(DEST_VAL_DIR, os.path.basename(image_right_path)))
        shutil.copy(disp_path, os.path.join(DEST_VAL_DIR, os.path.basename(disp_path)))
    print("Done")
    print("Saving the test images to the test directory...")
    for (image_left_path, image_right_path, disp_path) in tqdm(test_data_list, desc="test", unit="image"):
        shutil.copy(image_left_path, os.path.join(DEST_TEST_DIR, os.path.basename(image_left_path)))
        shutil.copy(image_right_path, os.path.join(DEST_TEST_DIR, os.path.basename(image_right_path)))
        shutil.copy(disp_path, os.path.join(DEST_TEST_DIR, os.path.basename(disp_path)))
    print("Done")
    print("Saving the split information to the split info file...")
    with open(os.path.join(DEST_ROOT, SPLIT_INFO_BASENAME), "w") as f:
        f.write(f"JAX tile train ratio: {JAX_TILE_TRAIN_RATIO}\n")
        f.write(f"JAX tile val ratio: {JAX_TILE_VAL_RATIO}\n")
        f.write(f"JAX tile test ratio: {JAX_TILE_TEST_RATIO}\n")
        f.write(f"OMA tile train ratio: {OMA_TILE_TRAIN_RATIO}\n")
        f.write(f"OMA tile val ratio: {OMA_TILE_VAL_RATIO}\n")
        f.write(f"OMA tile test ratio: {OMA_TILE_TEST_RATIO}\n")
        f.write(f"Train list length: {len(train_data_list)}\n")
        f.write(f"JAX train list length: {len(jax_train_data_list)}\n")
        f.write(f"OMA train list length: {len(oma_train_data_list)}\n")
        f.write(f"Val list length: {len(val_data_list)}\n")
        f.write(f"JAX val list length: {len(jax_val_data_list)}\n")
        f.write(f"OMA val list length: {len(oma_val_data_list)}\n")
        f.write(f"Test list length: {len(test_data_list)}\n")
        f.write(f"JAX test list length: {len(jax_test_data_list)}\n")
        f.write(f"OMA test list length: {len(oma_test_data_list)}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Train crop size: {TRAIN_CROP_SIZE}\n")
    print("Done")
    print("--------------------------------")
    print("The US3D dataset has been split successfully")
    print("--------------------------------")

if __name__ == "__main__":
    main()