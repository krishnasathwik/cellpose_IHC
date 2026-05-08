import tifffile
import numpy as np
import os
import random
from PIL import Image
import cv2

# -------- SETTINGS --------
image_path = r"C:\p53_tiff\0073\0073_20x_BF_01_raw.ome.btf"
patch_size = 512

num_patches_each = 10

output_folder = "MIB_patches"
normal_folder = os.path.join(output_folder, "normal")
blur_folder = os.path.join(output_folder, "blur")

white_threshold = 0.70
min_brown_pixels = 400
min_blue_pixels = 400

blur_threshold = 100  # adjust if needed
# --------------------------

os.makedirs(normal_folder, exist_ok=True)
os.makedirs(blur_folder, exist_ok=True)

print("Loading full slide into RAM...")
image = tifffile.imread(image_path)

height, width, channels = image.shape

# -------- FUNCTIONS --------

def detect_white(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return np.sum(gray > 220) / (patch_size * patch_size)

def detect_brown(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (10,40,20), (30,255,200))
    return np.sum(mask > 0)

def detect_blue(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (90,50,20), (140,255,255))
    return np.sum(mask > 0)

def compute_blur(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# -------- LOOP --------

normal_count = 0
blur_count = 0
attempts = 0

print("\nStarting patch extraction...\n")

while normal_count < num_patches_each or blur_count < num_patches_each:
    
    attempts += 1
    
    if attempts % 50 == 0:
        print(f"Attempts: {attempts} | Normal: {normal_count} | Blur: {blur_count}")
    
    x = random.randint(0, width - patch_size)
    y = random.randint(0, height - patch_size)
    
    patch = image[y:y+patch_size, x:x+patch_size]
    
    # ---- Tissue filtering ----
    if not (detect_white(patch) < white_threshold and
            detect_brown(patch) > min_brown_pixels and
            detect_blue(patch) > min_blue_pixels):
        continue
    
    blur_value = compute_blur(patch)
    
    # ---- NORMAL PATCH ----
    if blur_value > blur_threshold and normal_count < num_patches_each:
        normal_count += 1
        Image.fromarray(patch).save(f"{normal_folder}/normal_{normal_count}.png")
        print(f"✅ NORMAL {normal_count} | Blur={blur_value:.2f}")
    
    # ---- BLUR PATCH ----
    elif blur_value <= blur_threshold and blur_count < num_patches_each:
        blur_count += 1
        Image.fromarray(patch).save(f"{blur_folder}/blur_{blur_count}.png")
        print(f"🔵 BLUR {blur_count} | Blur={blur_value:.2f}")

print("\n🎉 DONE!")
print(f"Total attempts: {attempts}")