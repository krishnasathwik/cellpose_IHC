import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- INPUT PATHS --------
union_blur = "/kaggle/input/datasets/krishnasathwik12/union-images/blur_output"
union_normal = "/kaggle/input/datasets/krishnasathwik12/union-images/normal_output"

# -------- OUTPUT PATH --------
output_base = "/kaggle/working/final_ki67"
os.makedirs(output_base, exist_ok=True)

# -------- LOAD FILES --------
blur_files = glob.glob(os.path.join(union_blur, "*_grid.png"))
normal_files = glob.glob(os.path.join(union_normal, "*_grid.png"))

all_files = blur_files + normal_files

print(f"Found {len(blur_files)} blur + {len(normal_files)} normal = {len(all_files)} images")


# -------- UPDATED CLASSIFICATION FUNCTION --------
def classify_using_raw_and_union(raw_img_rgb, union_img_rgb, min_area=30, max_area=1500):

    raw_bgr = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2BGR)
    union_bgr = cv2.cvtColor(union_img_rgb, cv2.COLOR_RGB2BGR)

    # ---- Step 1: Extract nuclei mask from UNION ----
    b, g, r = cv2.split(union_bgr)
    red_mask = (r > 150) & (g < 100) & (b < 100)
    red_mask = red_mask.astype(np.uint8) * 255

    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- Step 2: Use RAW image for color ----
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)

    blue_count = 0
    brown_count = 0

    overlay = raw_bgr.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue

        mask = np.zeros(red_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        h = hsv[:, :, 0][mask == 255]
        s = hsv[:, :, 1][mask == 255]

        if len(h) == 0:
            continue

        mean_h = np.mean(h)
        mean_s = np.mean(s)

        # ---- Skip low saturation (noise/background) ----
        if mean_s < 10:
            continue

        # ---- Lenient Brown (Ki-67 +) ----
        if 0 < mean_h < 40 and mean_s > 20:
            brown_count += 1
            cv2.drawContours(overlay, [c], -1, (0, 0, 255), -1)

        # ---- Lenient Blue (Ki-67 -) ----
        elif 80 < mean_h < 160:
            blue_count += 1
            cv2.drawContours(overlay, [c], -1, (255, 0, 0), -1)

        # ---- Fallback (ENSURES NO LOSS) ----
        else:
            if abs(mean_h - 120) < abs(mean_h - 20):
                blue_count += 1
                cv2.drawContours(overlay, [c], -1, (255, 0, 0), -1)
            else:
                brown_count += 1
                cv2.drawContours(overlay, [c], -1, (0, 0, 255), -1)

    total = blue_count + brown_count
    ki67 = (brown_count / total) * 100 if total > 0 else 0

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay_rgb, blue_count, brown_count, ki67


# -------- PROCESS --------
for img_path in all_files:

    filename = os.path.basename(img_path)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img_rgb.shape

    # ---- Split 6-panel grid ----
    panel_w = w // 3
    panel_h = h // 2

    raw = img_rgb[0:panel_h, 0:panel_w]                      # Panel 1
    union = img_rgb[panel_h:2*panel_h, 2*panel_w:3*panel_w]  # Panel 6

    # ---- Classification ----
    classified_img, blue_cnt, brown_cnt, ki67 = classify_using_raw_and_union(raw, union)

    # ---- Create 3-panel output ----
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


print("\n✅ Done! Results saved in /kaggle/working/final_ki67/")