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
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
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
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Val ratio: {VAL_RATIO}")
    print(f"Test ratio: {TEST_RATIO}")
    print(f"Train crop size: {TRAIN_CROP_SIZE}")
    print("--------------------------------")
    print("Setting seed...")
    np.random.seed(SEED)
    random.seed(SEED)
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
    image_left_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, '*LEFT*.tif')))
    image_right_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, '*RIGHT*.tif')))
    disp_list = sorted(glob.glob(os.path.join(SOURCE_DISP_ROOT, '*DSP*.tif')))
    assert len(image_left_list) == len(image_right_list) == len(disp_list)
    print(f"Found {len(image_left_list)} images")
    print("Splitting data into train, val, and test sets...")
    data_list = list(zip(image_left_list, image_right_list, disp_list))
    train_list, val_list, test_list = random_split(data_list, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    print("Done")
    print("--------------------------------")
    print(f"Train split: {TRAIN_RATIO}, {len(train_list)} train images")
    print(f"Val split: {VAL_RATIO}, {len(val_list)} val images")
    print(f"Test split: {TEST_RATIO}, {len(test_list)} test images")
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
        f.write(f"Train ratio: {TRAIN_RATIO}\n")
        f.write(f"Val ratio: {VAL_RATIO}\n")
        f.write(f"Test ratio: {TEST_RATIO}\n")
        f.write(f"Train crop size: {TRAIN_CROP_SIZE}\n")
        f.write(f"Seed: {SEED}\n")
    print("Done")
    print("--------------------------------")
    print("The US3D dataset has been split successfully")
    print("--------------------------------")

if __name__ == "__main__":
    main()