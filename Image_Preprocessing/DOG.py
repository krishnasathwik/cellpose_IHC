import cv2
import numpy as np
import matplotlib.pyplot as plt

def blur_pipeline(img):
    img = img.astype(np.float32)

    # Difference of Gaussians (DoG) for detail extraction
    blur_small = cv2.GaussianBlur(img, (0, 0), 1.0)
    blur_large = cv2.GaussianBlur(img, (0, 0), 2.5)
    detail = blur_small - blur_large
    
    # Sharpening via detail injection
    img = img + 1.8 * detail

    # Boost Red channel (Index 2 is Red in RGB)
    img[:, :, 2] *= 1.2
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Contrast enhancement in LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Load 512x512 image
img_path = '/kaggle/input/datasets/krishnasathwik12/images-blur/blur_1.png'
img_bgr = cv2.imread(img_path)

# if img_bgr is not None:
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
#     # Process through pipeline
#     output_gray = blur_pipeline(img_rgb)
    
#     # Grayscale version of original for fair comparison
#     orig_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
#     # Display full 512x512 results
#     plt.figure(figsize=(14, 7))
    
#     plt.subplot(1, 2, 1)
#     plt.title('Original (Grayscale)')
#     plt.imshow(orig_gray, cmap='gray')
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     plt.title('Pipeline Output (Sharpened)')
#     plt.imshow(output_gray, cmap='gray')
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()
# ... (your existing blur_pipeline function) ...

if img_bgr is not None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    output_gray = blur_pipeline(img_rgb)
    
    # 1. Binary Thresholding (Essential for findContours)
    # Using OTSU to automatically find the best threshold value
    _, binary = cv2.threshold(output_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Draw Contours on a color version of the output so the lines are visible (e.g., Green)
    contour_img = cv2.cvtColor(output_gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2) # -1 draws all, (0,255,0) is Green, 2 is thickness

    # Display results
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.title('Sharpened Grayscale')
    plt.imshow(output_gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Contours Detected')
    plt.imshow(contour_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Total contours found: {len(contours)}")

else:
    print(f"Error: Could not find image at {img_path}")