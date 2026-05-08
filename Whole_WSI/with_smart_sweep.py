# ================================
#  WSI KI-67 PIPELINE (FINAL CORRECT)
# ================================

import openslide
import numpy as np
import cv2
import torch
from cellpose import models
import os, shutil, time, threading, queue
import tifffile as tiff

# ================================
# CONFIG
# ================================

wsi_path = "/kaggle/input/datasets/krishnasathwik12/0073-tiff/output_wsi.tiff"
final_wsi_path = "/kaggle/working/final_wsi_streamed.tiff"

tile_size = 1024
overlap = 256
stride = tile_size - overlap

batch_size = 12

blur_threshold = 100
min_nucleus_area = 30
max_nucleus_area = 1500
blue_boost = 15
margin = 64

TISSUE_LEVEL = 2
MORPH_CLOSE_KERNEL = 15
MIN_BLOB_AREA = 5000

# ================================
# GPU SETUP
# ================================

torch.backends.cudnn.benchmark = True

device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1")

model_dir = "/root/.cellpose/models"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

model_0 = models.Cellpose(gpu=True, device=device_0, model_type='nuclei')
model_1 = models.Cellpose(gpu=True, device=device_1, model_type='nuclei')

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

# ================================
# LOAD WSI + SMART SWEEP
# ================================

slide = openslide.OpenSlide(wsi_path)
W, H = slide.dimensions

downsample = slide.level_downsamples[TISSUE_LEVEL]
thumb = np.array(slide.read_region((0,0),TISSUE_LEVEL,slide.level_dimensions[TISSUE_LEVEL]))[:,:,:3]

gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
_, tissue_mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_CLOSE_KERNEL,MORPH_CLOSE_KERNEL))
dense_mask = cv2.morphologyEx(tissue_mask,cv2.MORPH_CLOSE,kernel)

contours,_ = cv2.findContours(dense_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
valid = [c for c in contours if cv2.contourArea(c)>=MIN_BLOB_AREA]

if not valid:
    boxes = [(0,0,W,H)]
else:
    boxes = [cv2.boundingRect(c) for c in valid]

def in_box(x,y):
    for bx,by,bw,bh in boxes:
        if bx*downsample <= x < (bx+bw)*downsample and by*downsample <= y < (by+bh)*downsample:
            return True
    return False

coords = [(x,y) for y in range(0,H,stride) for x in range(0,W,stride) if in_box(x,y)]

min_x = min(x for x,_ in coords)
min_y = min(y for _,y in coords)

coords = [((x,y), ((x-min_x)//tile_size, (y-min_y)//tile_size)) for x,y in coords]

tiles_x = max(tx for _,(tx,_) in coords)+1
tiles_y = max(ty for _,(_,ty) in coords)+1

# ================================
# FUNCTIONS (ALL PRESERVED)
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
    l = cv2.createCLAHE(1.5, (8, 8)).apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0.8)

def blur_pipeline(img):
    img = img.astype(np.float32)

    blur_small = cv2.GaussianBlur(img, (0, 0), 1.0)
    blur_large = cv2.GaussianBlur(img, (0, 0), 2.5)
    img = img + 1.8 * (blur_small - blur_large)

    img[:, :, 2] *= 1.2
    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def resize_mask(mask, shape):
    return cv2.resize(mask.astype(np.uint16), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

def get_geometric_contours(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_nucleus_area < area < max_nucleus_area):
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        circ = 4*np.pi*(area/(peri*peri))
        if circ < 0.45:
            continue

        hull = cv2.convexHull(c)
        if cv2.contourArea(hull) == 0:
            continue

        if area / cv2.contourArea(hull) < 0.75:
            continue

        valid.append(c)
    return valid

def get_unique_cv(contours, cp_mask):
    unique = []
    cp_bin = (cp_mask > 0).astype(np.uint8)

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        roi = c - [x,y]

        c_mask = np.zeros((h,w), np.uint8)
        cv2.drawContours(c_mask, [roi], -1, 1, -1)

        overlap = cv2.bitwise_and(c_mask, cp_bin[y:y+h, x:x+w])
        if np.sum(overlap) < 0.1 * cv2.contourArea(c):
            unique.append(c)

    return unique

def centroid_filter(mask):
    labels = np.unique(mask)
    labels = labels[labels != 0]

    new_mask = np.zeros_like(mask)

    for lbl in labels:
        coords = np.argwhere(mask == lbl)
        cy, cx = coords.mean(axis=0)

        if margin < cx < (mask.shape[1]-margin) and margin < cy < (mask.shape[0]-margin):
            new_mask[mask == lbl] = lbl

    return new_mask

def classify_and_draw(img, mask):
    ids = np.unique(mask)
    ids = ids[ids != 0]

    overlay = img.copy()
    blue = brown = 0

    for i in ids:
        m = (mask == i).astype(np.uint8)
        pixels = img[m == 1]

        if pixels.size == 0:
            continue

        if np.mean(pixels[:,0]) > np.mean(pixels[:,2]) + blue_boost:
            color = (0,0,255); brown += 1
        else:
            color = (255,0,0); blue += 1

        cnt,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnt, -1, color, 2)

    return overlay, blue, brown, len(ids)

# ================================
# QUEUES
# ================================

load_q = queue.Queue(64)
proc_q = queue.Queue(32)
write_q = queue.Queue(64)

total_blue=total_brown=total_nuclei=0
lock=threading.Lock()

# ================================
# LOADER
# ================================

def loader():
    for (coord,(tx,ty)) in coords:
        x,y=coord
        tile=np.array(slide.read_region((x,y),0,(tile_size,tile_size)))[:,:,:3]
        if np.mean(tile)>240: continue
        load_q.put((coord,tx,ty,tile))
    for _ in range(4): load_q.put(None)

# ================================
# PREPROCESS
# ================================

def preprocessor():
    batch=[]
    while True:
        item=load_q.get()
        if item is None: break

        coord,tx,ty,tile=item

        blur_score = compute_blur(tile)
        g = blur_pipeline(tile) if blur_score < blur_threshold else normal_pipeline(tile)

        batch.append((coord,tx,ty,tile,cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)))

        if len(batch)==batch_size:
            proc_q.put(batch)
            batch=[]

    if batch:
        proc_q.put(batch)

    proc_q.put(None)

# ================================
# GPU WORKER (FINAL CLEAN LOG)
# ================================

def gpu_worker(model, gpu_id):
    global total_blue,total_brown,total_nuclei

    print(f"🚀 GPU {gpu_id} started")

    batch_count = 0

    while True:
        batch=proc_q.get()

        if batch is None:
            proc_q.put(None)
            break

        batch_count += 1

        imgs=[b[4] for b in batch]

        infer_start = time.time()

        masks, _, _, _ = model.eval(imgs, channels=[[0,0]]*len(imgs))

        infer_end = time.time()

        print(f"[GPU {gpu_id}] Batch {batch_count} inference time: {infer_end - infer_start:.2f} sec")

        for (coord,tx,ty,tile,_),mask in zip(batch,masks):

            mask_cp = resize_mask(mask, tile.shape)

            contours = get_geometric_contours(cv2.cvtColor(imgs[batch.index((coord,tx,ty,tile,_))], cv2.COLOR_RGB2GRAY))
            unique_cv = get_unique_cv(contours, mask_cp)

            union = mask_cp.copy()
            max_id = np.max(union)

            for c in unique_cv:
                max_id += 1
                cv2.drawContours(union, [c], -1, int(max_id), -1)

            union = centroid_filter(union)

            overlay,b,br,tot = classify_and_draw(tile, union)

            write_q.put((ty,tx,overlay))

            with lock:
                total_blue+=b
                total_brown+=br
                total_nuclei+=tot

# ================================
# WRITER
# ================================

def writer():
    buffer={}
    row=0

    with tiff.TiffWriter(final_wsi_path,bigtiff=True) as tif:
        while True:
            item=write_q.get()
            if item is None: break

            ty,tx,tile=item
            buffer.setdefault(ty,{})[tx]=tile

            while row in buffer and len(buffer[row])==tiles_x:
                row_img=np.hstack([buffer[row][i] for i in range(tiles_x)])
                tif.write(row_img,compression='jpeg',photometric='rgb')
                del buffer[row]
                row+=1

# ================================
# RUN
# ================================

threads=[]

threads.append(threading.Thread(target=loader))

for _ in range(4):
    threads.append(threading.Thread(target=preprocessor))

threads.append(threading.Thread(target=gpu_worker,args=(model_0,0)))
threads.append(threading.Thread(target=gpu_worker,args=(model_1,1)))

threads.append(threading.Thread(target=writer))

for t in threads:
    t.start()

for t in threads[:-1]:
    t.join()

write_q.put(None)
threads[-1].join()

# ================================
# FINAL
# ================================

ki67=(total_brown/total_nuclei)*100 if total_nuclei else 0

print("\n===== FINAL =====")
print("Blue:",total_blue)
print("Brown:",total_brown)
print("Total:",total_nuclei)
print("Ki67:",round(ki67,2),"%")
print("Saved:",final_wsi_path)