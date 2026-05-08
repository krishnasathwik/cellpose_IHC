import os
import glob
import numpy as np
from cellpose import models
from PIL import Image
import cv2
import pandas as pd
import torch
import multiprocessing
import shutil
import time  # Added for inference timing

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

# Gather all file paths
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

print("\nLoading Cellpose nuclei model...")
try:
    model = models.CellposeModel(pretrained_model='nuclei', gpu=use_gpu)
    print("Cellpose model loaded with GPU =", use_gpu)
except Exception as e:
    print("GPU init failed, falling back to CPU.")
    print("Reason:", e)
    use_gpu = False
    model = models.CellposeModel(pretrained_model='nuclei', gpu=False)

# -------- FUNCTIONS --------
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

# -------- PROCESS DATA --------
print(f"\nFound {len(blur_files)} blur images and {len(normal_files)} normal images.")
print("Processing images and measuring inference time...")

results = []

for patch_path in all_patch_paths:
    filename = os.path.basename(patch_path)
    
    # --- FIX: ROBUST CATEGORY DETECTION ---
    # We check if the actual path string contains the blur directory name
    is_blur = "images-blur" in patch_path
    target_dir = out_blur if is_blur else out_normal
    category = "Blur" if is_blur else "Normal"

    # Load raw patch
    patch = np.array(Image.open(patch_path).convert("RGB")).astype(np.uint8)

    # 1. Custom Pipeline Processing
    blur_score = compute_blur(patch)
    if blur_score < blur_threshold:
        processed = blur_pipeline(patch)
    else:
        processed = normal_pipeline(patch)
    
    # --- TIMING CUSTOM INFERENCE ---
    start_custom = time.time()
    mask_custom, _, _ = model.eval(processed, diameter=20, cellprob_threshold=0.0, flow_threshold=0.4)
    end_custom = time.time()
    inf_time_custom = end_custom - start_custom

    # --- TIMING RAW INFERENCE ---
    start_raw = time.time()
    mask_raw, _, _ = model.eval(patch, diameter=20, cellprob_threshold=0.0, flow_threshold=0.4)
    end_raw = time.time()
    inf_time_raw = end_raw - start_raw

    # Resize masks if needed
    m_custom = resize_mask_to_image(mask_custom, patch.shape)
    m_raw = resize_mask_to_image(mask_raw, patch.shape)

    # Count nuclei
    count_custom = len(np.unique(m_custom)) - 1
    count_raw = len(np.unique(m_raw)) - 1

    # Store results
    results.append({
        "Category": category,
        "Filename": filename,
        "Blur_Score": blur_score,
        "Nuclei_Custom_Pipeline": count_custom,
        "Nuclei_Raw_Image": count_raw,
        "Difference": count_custom - count_raw,
        "Inference_Time_Custom_Sec": round(inf_time_custom, 4),
        "Inference_Time_Raw_Sec": round(inf_time_raw, 4),
        "Total_Inference_Time_Sec": round(inf_time_custom + inf_time_raw, 4)
    })

    # --- SAVE OVERLAYS ---
    overlay_custom = patch.copy()
    overlay_custom[m_custom > 0] = [255, 0, 0] # Red
    Image.fromarray(overlay_custom).save(os.path.join(target_dir, filename.replace(".png", "_custom_overlay.png")))

    overlay_raw = patch.copy()
    overlay_raw[m_raw > 0] = [0, 255, 0] # Green
    Image.fromarray(overlay_raw).save(os.path.join(target_dir, filename.replace(".png", "_raw_overlay.png")))

    print(f"Processed {filename} | Custom: {inf_time_custom:.2f}s | Raw: {inf_time_raw:.2f}s")

# -------- SAVE CSV --------
df = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "pipeline_comparison_results.csv")
df.to_csv(csv_path, index=False)

# -------- ZIP OUTPUT --------
print("\nZipping output files for download...")
shutil.make_archive('/kaggle/working/output_comparison_zip', 'zip', output_folder)

print("\n Done! Category bug is fixed and inference times are recorded.")