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
def classify_nuclei_rgb(raw_img_rgb, union_mask):
    """
    Classifies nuclei by averaging RGB pixels inside the mask and comparing 
    the Red vs Blue channel dominance, which works for all light/dark shades.
    """
    # ---- Connected Components (REAL nuclei separation) ----
    num_labels, labels = cv2.connectedComponents(union_mask.astype(np.uint8))

    blue_count = 0
    brown_count = 0
    overlay = raw_img_rgb.copy()

    # ---- Loop nuclei ----
    for label in range(1, num_labels):
        nucleus_mask = (labels == label)

        # Removed the area filter so EVERY mask gets classified as requested
        
        # 1. Get all RGB pixels for this specific nucleus
        pixels = raw_img_rgb[nucleus_mask]

        # 2. Calculate the average RGB value inside the mask [R, G, B]
        mean_rgb = np.mean(pixels, axis=0) 
        
        mean_r = mean_rgb[0]
        mean_b = mean_rgb[2]

        # 3. Dynamic Spectrum Logic:
        # Brown (DAB) naturally has higher Red than Blue.
        # Blue (Hematoxylin) naturally has higher Blue than Red.
        if mean_r > mean_b:
            brown_count += 1
            overlay[nucleus_mask] = [255, 0, 0]   # Assign RED for Brown
        else:
            blue_count += 1
            overlay[nucleus_mask] = [0, 0, 255]   # Assign BLUE for Blue

    total = blue_count + brown_count
    ki67 = (brown_count / total) * 100 if total > 0 else 0

    return overlay, blue_count, brown_count, ki67


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
    classified_img, blue_cnt, brown_cnt, ki67 = classify_nuclei_rgb(raw_rgb, union_mask)

    # ---- Visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(raw_rgb)
    axes[0].set_title("Raw Image")
    axes[0].axis('off')

    axes[1].imshow(union_mask, cmap='gray')
    axes[1].set_title("Union Mask")
    axes[1].axis('off')

    axes[2].imshow(classified_img)
    axes[2].set_title(f"Blue: {blue_cnt} | Brown: {brown_cnt} | Ki-67: {ki67:.2f}%")
    axes[2].axis('off')

    plt.tight_layout()

    save_name = filename.replace(".png", "_final.png")
    plt.savefig(os.path.join(output_base, save_name), dpi=150)
    plt.close()

    print(f"{filename} → Blue: {blue_cnt}, Brown: {brown_cnt}, Ki67: {ki67:.2f}%")

print("\n✅ DONE — RGB AVERAGING PIPELINE (SPECTRUM ADAPTIVE)")

# -------- ZIPPING EVERYTHING --------
print("\nZipping output directory...")
zip_path = "/kaggle/working/final_ki67_masks_results"
shutil.make_archive(zip_path, 'zip', output_base)

print(f"✅ All results are successfully zipped and ready to download at: {zip_path}.zip")