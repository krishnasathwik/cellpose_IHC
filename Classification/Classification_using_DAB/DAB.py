import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- INPUT PATHS --------
union_blur = "/kaggle/input/datasets/krishnasathwik12/union-images/blur_output"
union_normal = "/kaggle/input/datasets/krishnasathwik12/union-images/normal_output"

# -------- OUTPUT PATH --------
output_base = "/kaggle/working/final_ki67_DAB"
os.makedirs(output_base, exist_ok=True)

# -------- LOAD FILES --------
blur_files = glob.glob(os.path.join(union_blur, "*_grid.png"))
normal_files = glob.glob(os.path.join(union_normal, "*_grid.png"))

all_files = blur_files + normal_files

print(f"Found {len(blur_files)} blur + {len(normal_files)} normal = {len(all_files)} images")


# -------- STAIN DECONVOLUTION FUNCTION (FIXED + ADAPTIVE) --------
def classify_using_deconvolution(raw_img_rgb, union_img_rgb, min_area=30, max_area=1500):

    raw = raw_img_rgb.astype(np.float32) + 1
    union = union_img_rgb.copy()

    # ---- Step 1: Extract nuclei mask ----
    b, g, r = cv2.split(cv2.cvtColor(union, cv2.COLOR_RGB2BGR))
    red_mask = (r > 150) & (g < 100) & (b < 100)
    red_mask = red_mask.astype(np.uint8) * 255

    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- Step 2: Optical Density ----
    OD = -np.log(raw / 255.0)

    # ---- Step 3: Stain matrix ----
    stain_matrix = np.array([
        [0.65, 0.70, 0.29],  # Hematoxylin
        [0.27, 0.57, 0.78]   # DAB
    ])

    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)

    stain_inv = np.linalg.pinv(stain_matrix)

    # ---- Step 4: Deconvolution ----
    stains = np.dot(OD.reshape(-1,3), stain_inv)
    stains = stains.reshape(raw.shape[0], raw.shape[1], 2)

    dab = stains[:,:,1]

    # ---- Normalize DAB ----
    dab = (dab - dab.min()) / (dab.max() - dab.min() + 1e-8)

    # 🔥 ---- ADAPTIVE THRESHOLD (KEY FIX) ----
    dab_values_all = dab[red_mask == 255]

    if len(dab_values_all) > 0:
        threshold = np.percentile(dab_values_all, 60)  # tune: 55–70
    else:
        threshold = 0.5

    blue_count = 0
    brown_count = 0

    overlay = raw_img_rgb.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue

        mask = np.zeros(red_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        dab_vals = dab[mask == 255]

        if len(dab_vals) == 0:
            continue

        mean_dab = np.mean(dab_vals)

        # ---- Classification (ALL nuclei classified) ----
        if mean_dab > threshold:
            brown_count += 1
            cv2.drawContours(overlay, [c], -1, (255, 0, 0), -1)  # RED
        else:
            blue_count += 1
            cv2.drawContours(overlay, [c], -1, (0, 0, 255), -1)  # BLUE

    total = blue_count + brown_count
    ki67 = (brown_count / total) * 100 if total > 0 else 0

    return overlay, blue_count, brown_count, ki67


# -------- PROCESS --------
for img_path in all_files:

    filename = os.path.basename(img_path)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img_rgb.shape

    # ---- Split 6-panel grid ----
    panel_w = w // 3
    panel_h = h // 2

    raw = img_rgb[0:panel_h, 0:panel_w]
    union = img_rgb[panel_h:2*panel_h, 2*panel_w:3*panel_w]

    # ---- Classification ----
    classified_img, blue_cnt, brown_cnt, ki67 = classify_using_deconvolution(raw, union)

    # ---- Visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(raw)
    axes[0].set_title("Raw Image")
    axes[0].axis('off')

    axes[1].imshow(union)
    axes[1].set_title("Total Nuclei (Union)")
    axes[1].axis('off')

    axes[2].imshow(classified_img)
    axes[2].set_title(f"Blue: {blue_cnt} | Brown: {brown_cnt} | Ki-67: {ki67:.2f}%")
    axes[2].axis('off')

    plt.tight_layout()

    save_name = filename.replace("_grid.png", "_final.png")
    plt.savefig(os.path.join(output_base, save_name), dpi=150)
    plt.close()

    print(f"{filename} → Blue: {blue_cnt}, Brown: {brown_cnt}, Ki67: {ki67:.2f}%")


print("\n✅ Done! Results saved in /kaggle/working/final_ki67_DAB/")