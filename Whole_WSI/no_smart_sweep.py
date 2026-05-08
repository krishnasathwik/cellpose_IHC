# ================================
#  WSI KI-67 PIPELINE: EXACT KAGGLE LOGIC EDITION
# Architecture: CellposeModel Base + OpenCV Fallback + Full Slide Grid + Boundary Draw
# ================================

import os
# --- SPEED OPTIMIZATION 1: Disable Sparse Flow Overhead BEFORE Cellpose imports ---
os.environ["CELLPOSE_SPARSE_FLOW"] = "0"

import openslide
import numpy as np
import cv2
import torch
from cellpose import models
from concurrent.futures import ThreadPoolExecutor
import time
import queue
import threading
import tifffile
import multiprocessing

# --- SPEED OPTIMIZATION 2: PyTorch Hardware Benchmarking ---
torch.backends.cudnn.benchmark = True
physical_cores = multiprocessing.cpu_count() // 2
torch.set_num_threads(max(4, physical_cores))

# ================================
# -------- CONFIG --------
# ================================

wsi_path = "/home/pathousr6/output_wsi.tiff"
overlay_dir = "/home/pathousr6/output_overlays"
os.makedirs(overlay_dir, exist_ok=True)

tile_size = 512
overlap = 0          # ZERO OVERLAP for perfect grid drop-in
stride = tile_size
batch_size = 128     # A6000 can easily handle 128 512x512 tiles

# Exact Kaggle Thresholds
blur_threshold = 100
blue_boost = 15
cellpose_diameter = 20.0
min_nucleus_area = 30
max_nucleus_area = 1500

# ================================
# -------- GPU SETUP --------
# ================================

print("\n[INIT] CUDA available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Using base CellposeModel exactly like Kaggle
model = models.CellposeModel(pretrained_model='nuclei', gpu=torch.cuda.is_available(), device=device)
print(f"[INIT] Model loaded on {device}")

# ================================
# -------- GRID SETUP ----------
# ================================

total_start_time = time.time()
print("\n[GRID] Generating full slide coordinates...")

slide = openslide.OpenSlide(wsi_path)
W, H = slide.dimensions

# No Smart Sweep: Generate coordinates for the ENTIRE slide
raw_coords = [(x, y) for y in range(0, H, stride) for x in range(0, W, stride)]

total_batches = (len(raw_coords) + batch_size - 1) // batch_size
print(f"[GRID] Total 512x512 tiles to scan: {len(raw_coords)} | Max Batches: {total_batches}")

# ================================
# RAM CANVAS INITIALIZATION
# ================================
print("\n[STITCH] Initializing high-resolution composite canvas in RAM...")

canvas_width = W
canvas_height = H
print(f"[STITCH] Canvas size: {canvas_width} x {canvas_height} pixels")

print("[STITCH] Allocating memory block... (this takes a few seconds)")
canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
print("[STITCH] RAM Canvas initialized successfully.")

canvas_lock = threading.Lock()

# ================================
# -------- EXACT KAGGLE FUNCTIONS ------------
# ================================

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

def resize_mask(mask, shape):
    h, w = shape[:2]
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
        if not (min_area < area < max_area): continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < min_circularity: continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue

        solidity = area / hull_area
        if solidity < min_solidity: continue

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

def clear_border_nuclei(mask):
    """ Strictly excludes any nucleus touching the 1px edge of the tile """
    border_labels = set(mask[0, :]) | set(mask[-1, :]) | set(mask[:, 0]) | set(mask[:, -1])
    border_labels.discard(0) 

    for lbl in border_labels:
        mask[mask == lbl] = 0
    return mask

def classify_and_write(raw_tile, union_mask, coord):
    unique_ids = np.unique(union_mask)
    unique_ids = unique_ids[unique_ids != 0]

    blue_cnt, brown_cnt = 0, 0
    overlay = raw_tile.copy()

    for label in unique_ids:
        nucleus_mask = (union_mask == label)
        pixels_rgb = raw_tile[nucleus_mask]

        if len(pixels_rgb) == 0:
            continue

        mean_rgb = np.mean(pixels_rgb, axis=0)
        mean_r = mean_rgb[0]
        mean_b = mean_rgb[2]

        if mean_r > (mean_b + blue_boost):
            brown_cnt += 1
            color = (255, 0, 0)  # Red outline
        else:
            blue_cnt += 1
            color = (0, 0, 255)  # Blue outline

        nuc_contours, _ = cv2.findContours(nucleus_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, nuc_contours, -1, color, 2)

    x, y = coord
    h, w = overlay.shape[:2]
    
    # Trim to fit canvas bounds if on the absolute edge
    if y + h > canvas_height: h = canvas_height - y
    if x + w > canvas_width: w = canvas_width - x

    with canvas_lock:
        canvas[y:y+h, x:x+w] = overlay[:h, :w]

    return blue_cnt, brown_cnt, len(unique_ids)

# --- THREADED LOADER ---
def load_tile(coord):
    x, y = coord
    tile = slide.read_region((x, y), 0, (tile_size, tile_size))
    raw_tile = np.array(tile)[:, :, :3].astype(np.uint8)

    # Skip pure background immediately (saves massive GPU time on whole slide)
    if raw_tile.size == 0 or np.mean(raw_tile) > 230:
        return None

    blur_score = compute_blur(raw_tile)
    processed_gray = blur_pipeline(raw_tile) if blur_score < blur_threshold else normal_pipeline(raw_tile)
    processed_3c = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)

    return (coord, raw_tile, processed_3c, processed_gray)

# ================================
# GPU WORKER
# ================================
total_blue, total_brown, total_nuclei = 0, 0, 0
processed_batches = 0

batch_queue = queue.Queue(maxsize=15)
results_lock = threading.Lock()

def gpu_worker():
    global total_blue, total_brown, total_nuclei, processed_batches

    post_processor = ThreadPoolExecutor(max_workers=4)
    futures = []

    while True:
        tiles_batch = batch_queue.get()
        if tiles_batch is None:
            batch_queue.task_done()
            break

        inference_batch = [t[2] for t in tiles_batch]
        
        infer_start = time.time()

        # FULL 32-bit precision inference with exact Kaggle thresholds
        masks, _, _ = model.eval(
            inference_batch,
            diameter=cellpose_diameter,
            channels=[[0,0]] * len(inference_batch),
            cellprob_threshold=0.0,
            flow_threshold=0.4
        )
        infer_end = time.time()

        local_blue, local_brown, local_nuclei = 0, 0, 0

        for i, ((coord, raw_tile, _, processed_gray), mask_cp) in enumerate(zip(tiles_batch, masks)):
            mask_cp = resize_mask(mask_cp, raw_tile.shape)

            # ---- OpenCV Geometric Fallback ----
            geometric_contours = get_geometric_contours(processed_gray, min_nucleus_area, max_nucleus_area)
            unique_cv_contours = get_unique_cv_discoveries(geometric_contours, mask_cp)

            union_mask = mask_cp.copy()
            current_max_id = np.max(union_mask) if np.max(union_mask) > 0 else 0

            for c in unique_cv_contours:
                current_max_id += 1
                cv2.drawContours(union_mask, [c], -1, int(current_max_id), thickness=cv2.FILLED)

            # Drop cells touching the border to prevent half-drawn nuclei
            union_mask = clear_border_nuclei(union_mask)

            future = post_processor.submit(classify_and_write, raw_tile, union_mask, coord)
            futures.append(future)

        for f in futures:
            b, br, tot = f.result()
            local_blue += b
            local_brown += br
            local_nuclei += tot
        futures.clear()

        with results_lock:
            total_blue += local_blue
            total_brown += local_brown
            total_nuclei += local_nuclei
            processed_batches += 1
            print(f"[A6000] Processed Batch {processed_batches} | Speed: {infer_end - infer_start:.2f}s")

        batch_queue.task_done()

    post_processor.shutdown(wait=True)

# ================================
# MAIN EXECUTION
# ================================
worker = threading.Thread(target=gpu_worker)
worker.start()

tiles_batch = []
with ThreadPoolExecutor(max_workers=5) as loader:
    for result in loader.map(load_tile, raw_coords):
        if result is None:
            continue
        tiles_batch.append(result)
        if len(tiles_batch) == batch_size:
            batch_queue.put(tiles_batch)
            tiles_batch = []

if tiles_batch:
    batch_queue.put(tiles_batch)

batch_queue.put(None)
batch_queue.join()
worker.join()

# ================================
# GENERATE PYRAMIDAL TIFF
# ================================
tiff_path = "/home/pathousr6/wsi_highres_composite_pyramid.tiff"
print(f"\n[OUTPUT] Writing Pyramidal BigTIFF from RAM to {tiff_path}...")

tifffile.imwrite(
    tiff_path,
    canvas,
    bigtiff=True,
    photometric='rgb',
    tile=(256, 256),
    compression='zlib',
    resolution=(10000, 10000),
    software='FastPipeline'
)
print("[OUTPUT] Pyramidal TIFF successfully created.")

# ================================
# FINAL RESULTS
# ================================
ki67 = (total_brown / total_nuclei) * 100 if total_nuclei > 0 else 0
total_time = time.time() - total_start_time

print("\n===== FINAL WSI RESULT =====")
print(f"Blue (Negative) : {total_blue}")
print(f"Brown (Positive): {total_brown}")
print(f"Total Nuclei    : {total_nuclei}")
print(f"Ki-67 Index     : {ki67:.2f}%")
print(f"Time Taken      : {total_time/60:.2f} Minutes")