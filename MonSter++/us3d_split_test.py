import glob
import os
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm

SOURCE_IMAGES_ROOT = "/home/s0214/_scratch2/MonSter-plusplus/MonSter++/datasets/us3d/test"
DEST_DIR = SOURCE_IMAGES_ROOT
TEST_CROP_SIZE = (256, 1024)

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
    image_left_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, '*LEFT_RGB.tif')))
    image_right_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, '*RIGHT_RGB.tif')))
    dsp_list = sorted(glob.glob(os.path.join(SOURCE_IMAGES_ROOT, '*LEFT_DSP.tif')))
    print(f"Found {len(image_left_list)} images")
    print(f"Found {len(image_right_list)} images")
    print(f"Found {len(dsp_list)} images")
    assert len(image_left_list) == len(image_right_list) == len(dsp_list)
    test_data_list = list(zip(image_left_list, image_right_list, dsp_list))

    for (image_left_path, image_right_path, disp_path) in tqdm(test_data_list, desc="test", unit="image"):
        image_left = Image.open(image_left_path)
        image_right = Image.open(image_right_path)
        disp = Image.open(disp_path)
        cropped_left_images = get_cropped_array_list_by_size(np.array(image_left), TEST_CROP_SIZE)
        cropped_right_images = get_cropped_array_list_by_size(np.array(image_right), TEST_CROP_SIZE)
        cropped_disp_images = get_cropped_array_list_by_size(np.array(disp, dtype=np.float32), TEST_CROP_SIZE)
        if cropped_left_images is not None and cropped_right_images is not None and cropped_disp_images is not None:
            for i, (cropped_left_image, cropped_right_image, cropped_disp_image) in enumerate(zip(cropped_left_images, cropped_right_images, cropped_disp_images)):
                left_image_path = os.path.join(DEST_DIR, add_suffix_before_extension(os.path.basename(image_left_path), f"_{i}"))
                right_image_path = os.path.join(DEST_DIR, add_suffix_before_extension(os.path.basename(image_right_path), f"_{i}"))
                disp_image_path = os.path.join(DEST_DIR, add_suffix_before_extension(os.path.basename(disp_path), f"_{i}"))
                Image.fromarray(cropped_left_image).convert("RGB").save(left_image_path)
                Image.fromarray(cropped_right_image).convert("RGB").save(right_image_path)
                Image.fromarray(cropped_disp_image).convert("F").save(disp_image_path)
        os.remove(image_left_path)
        os.remove(image_right_path)
        os.remove(disp_path)

if __name__ == "__main__":
    main()