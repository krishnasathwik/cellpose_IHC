# (same imports as your original)
import os
import glob
import numpy as np
from cellpose import models
from PIL import Image
import cv2
import torch
import multiprocessing
import matplotlib.pyplot as plt
import shutil

# -------- GPU CHECK --------
print("Torch version:", torch.__version__)
print("CUDA version :", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name     :", torch.cuda.get_device_name(0))

use_gpu = torch.cuda.is_available()

# -------- CPU OPTIMIZATION --------
physical_cores = multiprocessing.cpu_count() // 2
torch.set_num_threads(max(4, physical_cores))

# -------- DIRECTORIES --------
dir_blur = "/kaggle/input/datasets/krishnasathwik12/images-blur"
dir_normal = "/kaggle/input/datasets/krishnasathwik12/images-normal"

blur_files = glob.glob(os.path.join(dir_blur, "*.png"))
normal_files = glob.glob(os.path.join(dir_normal, "*.png"))
all_patch_paths = blur_files + normal_files

# -------- OUTPUT FOLDERS --------
output_base = "/kaggle/working/output"
blur_output_dir = os.path.join(output_base, "blur_output")
normal_output_dir = os.path.join(output_base, "normal_output")

os.makedirs(blur_output_dir, exist_ok=True)
os.makedirs(normal_output_dir, exist_ok=True)

# -------- NEW MASK OUTPUT --------
mask_base = "/kaggle/working/output_masks"
mask_blur_dir = os.path.join(mask_base, "blur_output")
mask_normal_dir = os.path.join(mask_base, "normal_output")

os.makedirs(mask_blur_dir, exist_ok=True)
os.makedirs(mask_normal_dir, exist_ok=True)

# -------- PARAMETERS --------
blur_threshold = 100
min_nucleus_area = 30
max_nucleus_area = 1500

print("\nLoading Cellpose nuclei model...")
model = models.CellposeModel(pretrained_model='nuclei', gpu=use_gpu)

# -------- FUNCTIONS (UNCHANGED) --------
def compute_blur(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def normal_pipeline(img):
    img = img.astype(np.float32)
    img[:, :, 2] *= 1.1
    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0.8)

def blur_pipeline(img):
    img = img.astype(np.float32)

    blur_small = cv2.GaussianBlur(img, (0, 0), 1.0)
    blur_large = cv2.GaussianBlur(img, (0, 0), 2.5)
    detail = blur_small - blur_large
    img = img + 1.8 * detail

    img[:, :, 2] *= 1.2
    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def resize_mask_to_image(mask, image_shape):
    h, w = image_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

def get_geometric_contours(gray_img, min_area, max_area, min_circularity=0.45, min_solidity=0.75):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area
        if solidity < min_solidity:
            continue

        valid_contours.append(c)

    return valid_contours

def get_unique_cv_discoveries(cv_contours, cp_mask):
    unique_contours = []
    cp_binary = (cp_mask > 0).astype(np.uint8)

    for c in cv_contours:
        x, y, w, h = cv2.boundingRect(c)
        roi_contour = c - [x, y]

        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [roi_contour], -1, 1, thickness=cv2.FILLED)

        cp_roi = cp_binary[y:y+h, x:x+w]
        overlap = cv2.bitwise_and(c_mask, cp_roi)

        if np.sum(overlap) < (0.10 * cv2.contourArea(c)):
            unique_contours.append(c)

    return unique_contours


# -------- PROCESS --------
for patch_path in all_patch_paths:

    filename = os.path.basename(patch_path)
    is_blur_image = patch_path in blur_files

    target_dir = blur_output_dir if is_blur_image else normal_output_dir
    mask_dir = mask_blur_dir if is_blur_image else mask_normal_dir

    patch = np.array(Image.open(patch_path).convert("RGB")).astype(np.uint8)

    blur_score = compute_blur(patch)
    processed_gray = blur_pipeline(patch) if blur_score < blur_threshold else normal_pipeline(patch)

    processed_3c = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)

    # ---- Cellpose ----
    mask_preproc, _, _ = model.eval(processed_3c, diameter=20)
    mask_preproc = resize_mask_to_image(mask_preproc, patch.shape)

    # ---- OpenCV ----
    geometric_contours = get_geometric_contours(processed_gray, min_nucleus_area, max_nucleus_area)
    unique_cv_contours = get_unique_cv_discoveries(geometric_contours, mask_preproc)

    # ===============================
    # 🔥 FIXED: CREATE INSTANCE UNION MASK
    # ===============================
    # 1. Start with a copy of the Cellpose instance mask (preserves unique IDs 1, 2, 3...)
    union_mask = mask_preproc.copy()
    
    # 2. Find the highest ID currently used by Cellpose
    current_max_id = np.max(union_mask)
    
    # 3. Add each OpenCV contour with a BRAND NEW unique ID
    for c in unique_cv_contours:
        current_max_id += 1
        cv2.drawContours(union_mask, [c], -1, int(current_max_id), thickness=cv2.FILLED)

    # ---- SAVE MASK ----
    np.save(os.path.join(mask_dir, filename.replace(".png", "_mask.npy")), union_mask)