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

# -------- PARAMETERS --------
blur_threshold = 100
min_nucleus_area = 30
max_nucleus_area = 1500

print("\nLoading Cellpose nuclei model...")
try:
    model = models.CellposeModel(pretrained_model='nuclei', gpu=use_gpu)
    print("Cellpose model loaded with GPU =", use_gpu)
except Exception as e:
    print("GPU init failed, falling back to CPU.")
    print("Reason:", e)
    use_gpu = False
    model = models.CellposeModel(pretrained_model='nuclei', gpu=False)

# -------- PIPELINE FUNCTIONS --------
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

# -------- OPENCV GEOMETRIC FILTERING & UNION LOGIC --------
def get_geometric_contours(gray_img, min_area, max_area, min_circularity=0.45, min_solidity=0.75):
    """Finds contours and mathematically drops jagged/hollow background blobs."""
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
    """Returns only the OpenCV contours that Cellpose missed."""
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

# -------- PROCESS DATA --------
print(f"\nFound {len(blur_files)} blur images and {len(normal_files)} normal images.")
print("Running pipeline and generating 6‑panel grids...")

for patch_path in all_patch_paths:
    filename = os.path.basename(patch_path)
    
    # Determine which folder this image belongs to
    is_blur_image = patch_path in blur_files
    target_dir = blur_output_dir if is_blur_image else normal_output_dir

    # 1. Raw input image (RGB)
    patch = np.array(Image.open(patch_path).convert("RGB")).astype(np.uint8)

    # 2. Preprocessed image (grayscale)
    blur_score = compute_blur(patch)
    if blur_score < blur_threshold:
        processed_gray = blur_pipeline(patch)
    else:
        processed_gray = normal_pipeline(patch)

    # Preprocessed image as 3‑channel (for Cellpose input)
    processed_3c = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)

    # ---- Run Cellpose on raw image (Panel 3) ----
    mask_raw, _, _ = model.eval(
        patch,                         # raw RGB image
        diameter=20,
        cellprob_threshold=0.0,
        flow_threshold=0.4
    )
    mask_raw = resize_mask_to_image(mask_raw, patch.shape)
    raw_count = len(np.unique(mask_raw)) - 1

    # ---- Run Cellpose on preprocessed image (Panel 4) ----
    mask_preproc, _, _ = model.eval(
        processed_3c,                  # 3‑channel preprocessed image
        diameter=20,
        cellprob_threshold=0.0,
        flow_threshold=0.4
    )
    mask_preproc = resize_mask_to_image(mask_preproc, patch.shape)
    preproc_count = len(np.unique(mask_preproc)) - 1

    # ---- OpenCV geometric contours on preprocessed image (Panel 5) ----
    geometric_contours = get_geometric_contours(
        processed_gray, min_nucleus_area, max_nucleus_area
    )
    geometric_count = len(geometric_contours)

    # ---- Union for Panel 6 ----
    unique_cv_contours = get_unique_cv_discoveries(geometric_contours, mask_preproc)
    union_count = preproc_count + len(unique_cv_contours)

    # ---------- Create overlays (all on raw image, masks in red) ----------
    # All overlays will be drawn on the raw image (patch) in BGR, then converted to RGB for matplotlib.
    # We'll create separate copies for each panel to avoid overwriting.

    # Panel 3: raw image + mask_raw (red)
    overlay_raw = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    overlay_raw[mask_raw > 0] = [0, 0, 255]       # BGR red

    # Panel 4: raw image + mask_preproc (red)
    overlay_raw_preproc = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    overlay_raw_preproc[mask_preproc > 0] = [0, 0, 255]

    # Panel 5: raw image + geometric_contours (red)
    overlay_raw_geometric = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_raw_geometric, geometric_contours, -1, (0, 0, 255), cv2.FILLED)

    # Panel 6: raw image + union (mask_preproc + unique_cv_contours) (red)
    overlay_raw_union = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    overlay_raw_union[mask_preproc > 0] = [0, 0, 255]
    cv2.drawContours(overlay_raw_union, unique_cv_contours, -1, (0, 0, 255), cv2.FILLED)

    # Convert all overlays to RGB for matplotlib
    overlay_raw_rgb = cv2.cvtColor(overlay_raw, cv2.COLOR_BGR2RGB)
    overlay_raw_preproc_rgb = cv2.cvtColor(overlay_raw_preproc, cv2.COLOR_BGR2RGB)
    overlay_raw_geometric_rgb = cv2.cvtColor(overlay_raw_geometric, cv2.COLOR_BGR2RGB)
    overlay_raw_union_rgb = cv2.cvtColor(overlay_raw_union, cv2.COLOR_BGR2RGB)

    # ---------- Create 6‑panel grid ----------
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax = axes.ravel()

    # Panel 1: Raw input image
    ax[0].imshow(patch)
    ax[0].set_title("1. Raw Input Image", fontsize=14)

    # Panel 2: Preprocessed image (grayscale)
    ax[1].imshow(processed_gray, cmap='gray')
    ax[1].set_title("2. Preprocessed Image", fontsize=14)

    # Panel 3: Raw image + Cellpose mask (from raw)
    ax[2].imshow(overlay_raw_rgb)
    ax[2].set_title(f"3. Raw + Cellpose (raw) (Count: {raw_count})", fontsize=14)

    # Panel 4: Raw image + Cellpose mask (from preprocessed)
    ax[3].imshow(overlay_raw_preproc_rgb)
    ax[3].set_title(f"4. Raw + Cellpose (preproc) (Count: {preproc_count})", fontsize=14)

    # Panel 5: Raw image + OpenCV geometric contours (from preprocessed)
    ax[4].imshow(overlay_raw_geometric_rgb)
    ax[4].set_title(f"5. Raw + OpenCV (preproc) (Count: {geometric_count})", fontsize=14)

    # Panel 6: Raw image + Final union mask
    ax[5].imshow(overlay_raw_union_rgb)
    ax[5].set_title(f"6. Final Union (Count: {union_count})", fontsize=14)

    for a in ax:
        a.axis('off')

    plt.tight_layout()

    # Save the grid
    save_filename = filename.replace(".png", "_grid.png")
    plt.savefig(os.path.join(target_dir, save_filename), bbox_inches='tight', dpi=150)
    plt.close(fig)  # free memory

    print(f"Processed {filename} | Saved to {os.path.basename(target_dir)} | Raw: {raw_count}, Preproc: {preproc_count}, Geometric: {geometric_count}, Union: {union_count}")

# -------- ZIPPING EVERYTHING --------
print("\nZipping output directories...")
zip_path = "/kaggle/working/pipeline_results"
shutil.make_archive(zip_path, 'zip', output_base)

print(f"✅ Done! All results are zipped and ready to download at: {zip_path}.zip")