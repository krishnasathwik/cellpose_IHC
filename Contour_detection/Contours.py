import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_path = '/kaggle/input/datasets/krishnasathwik12/images-blur/blur_1.png'
img_bgr = cv2.imread(img_path)

if img_bgr is not None:
    # Convert BGR to RGB for proper display with matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Convert original image to Grayscale 
    # (This is strictly for the thresholding step, as Otsu requires 1-channel images)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Binary Thresholding
    # Using OTSU to automatically find the best threshold value from the grayscale image
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Draw Contours directly on a copy of the original RGB patch
    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2) # (0,255,0) is Green, 2 is thickness

    # Display results
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Patch')
    plt.imshow(img_rgb)
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