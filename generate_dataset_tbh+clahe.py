import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import random

# CREATE DATASET DIRECTORIES
input_dataset_dir = '/home/arunitmaity/Projects/Lung_Segmentation/data'
jsrt_dir = os.path.join(input_dataset_dir, 'JSRT')  # PNG files directory
scr_left_masks_dir = os.path.join(input_dataset_dir, 'scr', 'left_mask')  # PNG files directory
scr_right_masks_dir = os.path.join(input_dataset_dir, 'scr', 'right_mask')  # PNG files directory
nlm_cxr_dir = os.path.join(input_dataset_dir, 'NLM-MontgomeryCXRSet', 'MontgomerySet', 'CXR_png')  # PNG files directory
nlm_left_masks_dir = os.path.join(input_dataset_dir, 'NLM-MontgomeryCXRSet', 'MontgomerySet', 'ManualMask',
                                  'leftMask')  # PNG files directory
nlm_right_masks_dir = os.path.join(input_dataset_dir, 'NLM-MontgomeryCXRSet', 'MontgomerySet', 'ManualMask',
                                   'rightMask')  # PNG files directory

final_data_dir = '/home/arunitmaity/Projects/Lung_Segmentation/processed_data/tbh+clahe/'
final_train_data_dir = os.path.join(final_data_dir,
                                    'train')  # Directory which has two subdirectories, 'cxrs' and 'masks'
final_test_data_dir = os.path.join(final_data_dir, 'test')  # Directory which has two subdirectories, 'cxrs' and 'masks'
final_val_data_dir = os.path.join(final_data_dir,
                                  'validation')  # Directory which has two subdirectories, 'cxrs' and 'masks'

nlm_cxrs_DIRS = glob(os.path.join(nlm_cxr_dir, '*.png'))
jsrt_cxrs_DIRS = glob(os.path.join(jsrt_dir, '*.png'))

# CREATE TRAIN, TEST and VALIDATION SPLIT
random.seed(420)
random.shuffle(nlm_cxrs_DIRS)
random.shuffle(jsrt_cxrs_DIRS)
nlm_train = nlm_cxrs_DIRS[0:110]
nlm_test = nlm_cxrs_DIRS[110:124]
nlm_val = nlm_cxrs_DIRS[124:]
jsrt_train = jsrt_cxrs_DIRS[0:197]
jsrt_test = jsrt_cxrs_DIRS[197:222]
jsrt_val = jsrt_cxrs_DIRS[222:]

for image_file in tqdm(nlm_cxrs_DIRS):
    base_file = os.path.basename(image_file)
    left_mask_file = os.path.join(nlm_left_masks_dir, base_file)
    right_mask_file = os.path.join(nlm_right_masks_dir, base_file)

    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    left_mask = cv2.imread(left_mask_file, cv2.IMREAD_GRAYSCALE)
    right_mask = cv2.imread(right_mask_file, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (512, 512))
    left_mask = cv2.resize(left_mask, (512, 512))
    right_mask = cv2.resize(right_mask, (512, 512))
    mask = np.maximum(left_mask, right_mask)

    filterSize = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)
    tophat_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat_img = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    image = image + tophat_img - blackhat_img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    if image_file in nlm_train:
        cv2.imwrite(os.path.join(final_train_data_dir, 'cxrs', base_file), image)
        cv2.imwrite(os.path.join(final_train_data_dir, 'masks', base_file), mask)
    if image_file in nlm_val:
        cv2.imwrite(os.path.join(final_val_data_dir, 'cxrs', base_file), image)
        cv2.imwrite(os.path.join(final_val_data_dir, 'masks', base_file), mask)
    if image_file in nlm_test:
        cv2.imwrite(os.path.join(final_test_data_dir, 'cxrs', base_file), image)
        cv2.imwrite(os.path.join(final_test_data_dir, 'masks', base_file), mask)

for image_file in tqdm(jsrt_cxrs_DIRS):
    base_file = os.path.basename(image_file)
    left_mask_file = os.path.join(scr_left_masks_dir, base_file)
    right_mask_file = os.path.join(scr_right_masks_dir, base_file)

    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    left_mask = cv2.imread(left_mask_file, cv2.IMREAD_GRAYSCALE)
    right_mask = cv2.imread(right_mask_file, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (512, 512))
    left_mask = cv2.resize(left_mask, (512, 512))
    right_mask = cv2.resize(right_mask, (512, 512))
    mask = np.maximum(left_mask, right_mask)

    filterSize = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)
    tophat_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat_img = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    image = image + tophat_img - blackhat_img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    if image_file in jsrt_train:
        cv2.imwrite(os.path.join(final_train_data_dir, 'cxrs', base_file), image)
        cv2.imwrite(os.path.join(final_train_data_dir, 'masks', base_file), mask)
    if image_file in jsrt_val:
        cv2.imwrite(os.path.join(final_val_data_dir, 'cxrs', base_file), image)
        cv2.imwrite(os.path.join(final_val_data_dir, 'masks', base_file), mask)
    if image_file in jsrt_test:
        cv2.imwrite(os.path.join(final_test_data_dir, 'cxrs', base_file), image)
        cv2.imwrite(os.path.join(final_test_data_dir, 'masks', base_file), mask)
