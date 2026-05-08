import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# -------- STAIN DECONVOLUTION --------
def get_dab_channel(raw_img):

    raw = raw_img.astype(np.float32) + 1
    OD = -np.log(raw / 255.0)

    # H + DAB stain matrix
    stain_matrix = np.array([
        [0.65, 0.70, 0.29],  # Hematoxylin
        [0.27, 0.57, 0.78]   # DAB
    ])

    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)

    stain_inv = np.linalg.pinv(stain_matrix)

    stains = np.dot(OD.reshape(-1, 3), stain_inv)
    stains = stains.reshape(raw.shape[0], raw.shape[1], 2)

    dab = stains[:, :, 1]

    # normalize
    dab = (dab - dab.min()) / (dab.max() - dab.min() + 1e-8)

    return dab


# -------- CLASSIFICATION --------
def classify_nuclei(raw_img, union_mask):

    # ---- Connected Components (REAL nuclei separation) ----
    num_labels, labels = cv2.connectedComponents(union_mask.astype(np.uint8))

    dab = get_dab_channel(raw_img)

    # 🔥 Adaptive threshold
    dab_values = dab[union_mask > 0]
    threshold = np.percentile(dab_values, 60) if len(dab_values) > 0 else 0.5

    blue_count = 0
    brown_count = 0

    overlay = raw_img.copy()

    # ---- Loop nuclei ----
    for label in range(1, num_labels):

        nucleus_mask = (labels == label)

        area = np.sum(nucleus_mask)
        if area < 30 or area > 1500:
            continue

        mean_dab = np.mean(dab[nucleus_mask])

        # ---- FORCE classification (no skipping) ----
        if mean_dab > threshold:
            brown_count += 1
            overlay[nucleus_mask] = [255, 0, 0]   # RED
        else:
            blue_count += 1
            overlay[nucleus_mask] = [0, 0, 255]   # BLUE

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
    classified_img, blue_cnt, brown_cnt, ki67 = classify_nuclei(raw_rgb, union_mask)

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

print("\n✅ DONE — BEST PIPELINE (MASK + DAB)")