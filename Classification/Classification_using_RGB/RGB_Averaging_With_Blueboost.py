import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

# -------- PATHS --------
raw_blur = "/kaggle/input/datasets/krishnasathwik12/images-blur"
raw_normal = "/kaggle/input/datasets/krishnasathwik12/images-normal"

mask_blur = "/kaggle/working/output_masks/blur_output"
mask_normal = "/kaggle/working/output_masks/normal_output"

output_base = "/kaggle/working/final_ki67_masks"
os.makedirs(output_base, exist_ok=True)

# -------- LOAD FILES --------
raw_files = glob.glob(os.path.join(raw_blur, "*.png")) + \
            glob.glob(os.path.join(raw_normal, "*.png"))

print(f"Found {len(raw_files)} raw images")

# -------- CLASSIFICATION (STRICTLY RGB - SPECTRUM ADAPTIVE) --------
def classify_nuclei_rgb(raw_img_rgb, union_mask, blue_boost=15):
    """
    Classifies nuclei by averaging RGB pixels inside the mask.
    Because union_mask is already an Instance Mask with unique IDs, 
    we bypass connectedComponents.
    """
    # 1. Get all unique nucleus IDs (ignoring 0, which is the background)
    unique_ids = np.unique(union_mask)
    unique_ids = unique_ids[unique_ids != 0]
    
    total_masks = len(unique_ids)

    blue_count = 0
    brown_count = 0
    overlay = raw_img_rgb.copy()

    # ---- Loop nuclei directly using their unique IDs ----
    for label in unique_ids:
        # Isolate exactly one nucleus
        nucleus_mask = (union_mask == label)
        
        # 1. Get all RGB pixels for this specific nucleus
        pixels = raw_img_rgb[nucleus_mask]

        # Prevent crash if a mask somehow has 0 area
        if len(pixels) == 0:
            continue

        # 2. Calculate the average RGB value inside the mask [R, G, B]
        mean_rgb = np.mean(pixels, axis=0) 
        
        mean_r = mean_rgb[0]
        mean_b = mean_rgb[2]

        # 3. Dynamic Spectrum Logic with Blue Boost:
        # Require the Red channel to be higher than the Blue channel + the boost margin.
        # If it only wins by a slight margin, it gets classified as Blue.
        if mean_r > (mean_b + blue_boost):
            brown_count += 1
            overlay[nucleus_mask] = [255, 0, 0]   # Assign RED for Brown
        else:
            blue_count += 1
            overlay[nucleus_mask] = [0, 0, 255]   # Assign BLUE for Blue

    # Calculate Ki-67 (sanity check: total should exactly equal blue_count + brown_count)
    ki67 = (brown_count / total_masks) * 100 if total_masks > 0 else 0

    return overlay, blue_count, brown_count, total_masks, ki67


# -------- PROCESS --------
for raw_path in raw_files:
    filename = os.path.basename(raw_path)

    # ---- Load RAW ----
    raw = cv2.imread(raw_path)
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # ---- Match mask path ----
    if "blur" in raw_path:
        mask_path = os.path.join(mask_blur, filename.replace(".png", "_mask.npy"))
    else:
        mask_path = os.path.join(mask_normal, filename.replace(".png", "_mask.npy"))

    if not os.path.exists(mask_path):
        print("Missing mask:", filename)
        continue

    union_mask = np.load(mask_path)

    # ---- Classification ----
    # You can tweak the 15 here if you need more or less handicap for Blue
    classified_img, blue_cnt, brown_cnt, total_masks, ki67 = classify_nuclei_rgb(raw_rgb, union_mask, blue_boost=15)

    # ---- Visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    axes[0].imshow(raw_rgb)
    axes[0].set_title("Raw Image")
    axes[0].axis('off')

    # Note: the raw union mask looks weird using plt.imshow directly 
    # because the IDs are thousands, but it works mathematically!
    axes[1].imshow(union_mask > 0, cmap='gray') 
    axes[1].set_title(f"Union Mask (Total: {total_masks})")
    axes[1].axis('off')

    axes[2].imshow(classified_img)
    axes[2].set_title(f"Blue: {blue_cnt} | Brown: {brown_cnt} | Total: {total_masks} | Ki-67: {ki67:.2f}%")
    axes[2].axis('off')

    plt.tight_layout()

    save_name = filename.replace(".png", "_final.png")
    plt.savefig(os.path.join(output_base, save_name), dpi=150)
    plt.close()

    print(f"{filename} → Blue: {blue_cnt}, Brown: {brown_cnt}, Total: {total_masks}, Ki67: {ki67:.2f}%")

print("\n✅ DONE — RGB AVERAGING PIPELINE (SPECTRUM ADAPTIVE WITH BLUE BOOST)")

# -------- ZIPPING EVERYTHING --------
print("\nZipping output directory...")
zip_path = "/kaggle/working/final_ki67_masks_results"
shutil.make_archive(zip_path, 'zip', output_base)

print(f"✅ All results are successfully zipped and ready to download at: {zip_path}.zip")