import os
import glob
import numpy as np
from cellpose import models
from PIL import Image
import cv2
import torch
import multiprocessing
import shutil
import time

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
output_folder = "/kaggle/working/output"
out_blur = os.path.join(output_folder, "output_blur")
out_normal = os.path.join(output_folder, "output_normal")

os.makedirs(out_blur, exist_ok=True)
os.makedirs(out_normal, exist_ok=True)

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
    
    # FIX: Added the cv2. prefix here
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
    
    kernel = np.ones((3,3), np.uint8)
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
            
        # Circularity: 1.0 is a perfect circle. Tissues/Stains are usually < 0.3.
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue
            
        # Solidity: Ratio of contour area to its convex hull area. Drops "C" or "donut" shapes.
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
        
        # If the overlap is LESS than 10%, Cellpose missed it entirely. 
        # OpenCV claims this as a Unique Discovery.
        if np.sum(overlap) < (0.10 * cv2.contourArea(c)):
            unique_contours.append(c)

    return unique_contours

# -------- PROCESS DATA --------
print(f"\nFound {len(blur_files)} blur images and {len(normal_files)} normal images.")
print("Running Ensemble Union Pipeline...")

for patch_path in all_patch_paths:
    filename = os.path.basename(patch_path)
    
    is_blur = "images-blur" in patch_path
    target_dir = out_blur if is_blur else out_normal

    patch = np.array(Image.open(patch_path).convert("RGB")).astype(np.uint8)

    # 1. Custom Preprocessing
    blur_score = compute_blur(patch)
    if blur_score < blur_threshold:
        processed_gray = blur_pipeline(patch)
    else:
        processed_gray = normal_pipeline(patch)
        
    cv2.imwrite(os.path.join(target_dir, filename.replace(".png", "_preprocessed.png")), processed_gray)

    # 2. Pipeline Cellpose (The Base)
    processed_3c = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
    mask_cp_pipeline, _, _ = model.eval(processed_3c, diameter=20, cellprob_threshold=0.0, flow_threshold=0.4)
    mask_cp_pipeline = resize_mask_to_image(mask_cp_pipeline, patch.shape)
    cp_pipeline_count = len(np.unique(mask_cp_pipeline)) - 1

    # 3. OpenCV Geometric Contours & Union Check
    raw_geometric_contours = get_geometric_contours(processed_gray, min_nucleus_area, max_nucleus_area)
    unique_cv_contours = get_unique_cv_discoveries(raw_geometric_contours, mask_cp_pipeline)
    
    # 4. Final Union Score
    total_union_count = cp_pipeline_count + len(unique_cv_contours)

    # 5. Raw Patch -> Cellpose (Strictly for Baseline Comparison)
    mask_raw, _, _ = model.eval(patch, diameter=20, cellprob_threshold=0.0, flow_threshold=0.4)
    m_raw = resize_mask_to_image(mask_raw, patch.shape)
    baseline_raw_count = len(np.unique(m_raw)) - 1

    # --- CREATE OVERLAYS AND ADD TEXT ---
    
    # A. Ensemble Union Overlay
    overlay_union = cv2.cvtColor(patch.copy(), cv2.COLOR_RGB2BGR)
    
    # Draw Cellpose Base in solid RED (BGR format: [0, 0, 255])
    overlay_union[mask_cp_pipeline > 0] = [0, 0, 255]
    
    # Draw OpenCV Unique Discoveries in solid RED (cv2.FILLED fills the contours completely)
    cv2.drawContours(overlay_union, unique_cv_contours, -1, (0, 0, 255), cv2.FILLED) 
    
    text_union = f"Union Pipeline: {total_union_count} | Raw Baseline: {baseline_raw_count}"
    cv2.putText(overlay_union, text_union, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(overlay_union, text_union, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(target_dir, filename.replace(".png", "_pipeline_union_overlay.png")), overlay_union)

    # B. Raw Cellpose Baseline
    overlay_raw = patch.copy()
    overlay_raw[m_raw > 0] = [255, 0, 0] 
    overlay_raw_bgr = cv2.cvtColor(overlay_raw, cv2.COLOR_RGB2BGR)
    
    text_raw = f"Raw(Cellpose): {baseline_raw_count}"
    cv2.putText(overlay_raw_bgr, text_raw, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(overlay_raw_bgr, text_raw, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(target_dir, filename.replace(".png", "_raw_overlay.png")), overlay_raw_bgr)

    print(f"Processed {filename} | CP Base: {cp_pipeline_count} + CV Unique: {len(unique_cv_contours)} = {total_union_count} | Baseline: {baseline_raw_count}")

# -------- ZIP OUTPUT --------
print("\nZipping output files for download...")
shutil.make_archive('/kaggle/working/output_comparison_zip', 'zip', output_folder)

print("\n✅ Done! Ensemble Union Pipeline executed.")