import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
 
# ── PATHS  ───────────────────────────
base = os.path.expanduser('~/Downloads/spr26/cs191w')
original_path = os.path.join(base, 'images', 'original.png')
masks_folder = os.path.join(base, 'images', 'masks')
depth_folder  = os.path.join(base, 'images', 'depth_outputs')
figures_folder = os.path.join(base, 'images', 'figures')
 
# ── HELPER ─────────────────────────────────────────
def add_label(img_bgr, text, font_scale=1.2, thickness=2):
    out = img_bgr.copy()
    cv2.putText(out, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0,0,0), thickness+4, cv2.LINE_AA)
    cv2.putText(out, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return out
 
def resize_to(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
 
TARGET_W = 600
TARGET_H = 340
PAD = 8
BG = 240  # light grey background
 
# ── FIGURE 1: Original image ───────────────────────────────────────────
print("Making Figure 1: original image...")
orig = cv2.imread(original_path)
orig_resized = resize_to(orig, TARGET_W * 3 + PAD * 2, TARGET_H)
orig_labeled = add_label(orig_resized, "Input Image")
cv2.imwrite(os.path.join(figures_folder, 'fig1_original.png'), orig_labeled)
print("  Saved fig1_original.png")
 
# ── FIGURE 2: Depth map comparison (3 models side by side) ────────────
print("Making Figure 2: depth map comparison...")
depth_files = {
    'Depth Anything V2': os.path.join(depth_folder, 'DepthAnythingV2.png'),
    'MiDaS':             os.path.join(depth_folder, 'MiDaSDepthMap.png'),
    'ZoeDepth':          os.path.join(depth_folder, 'ZoeDepthMap.png'),
}
 
depth_imgs = []
for label, path in depth_files.items():
    img = cv2.imread(path)
    if img is None:
        print(f"  WARNING: could not load {path}")
        img = np.ones((TARGET_H, TARGET_W, 3), dtype=np.uint8) * 128
    img = resize_to(img, TARGET_W, TARGET_H)
    img = add_label(img, label)
    depth_imgs.append(img)
 
pad = np.ones((TARGET_H, PAD, 3), dtype=np.uint8) * BG
fig2 = np.hstack([depth_imgs[0], pad, depth_imgs[1], pad, depth_imgs[2]])
cv2.imwrite(os.path.join(figures_folder, 'fig2_depth_comparison.png'), fig2)
print("  Saved fig2_depth_comparison.png")
 
# ── FIGURE 3: Layer mask grid (3 models x 3 layers) ───────────────────
print("Making Figure 3: layer mask grid...")
models = ['DepthAnythingV2', 'MiDaSDepthMap', 'ZoeDepthMap']
model_labels = ['Depth Anything V2', 'MiDaS', 'ZoeDepth']
layer_labels = ['Layer 0 (BG)', 'Layer 1 (MG)', 'Layer 2 (FG)']
 
MASK_W = 460
MASK_H = 260
pad_v = np.ones((PAD, MASK_W * 3 + PAD * 2, 3), dtype=np.uint8) * BG
pad_h = np.ones((MASK_H, PAD, 3), dtype=np.uint8) * BG
 
rows = []
for mi, (model, mlabel) in enumerate(zip(models, model_labels)):
    row_imgs = []
    for li in range(3):
        mask_name = f'{model}_layer_{li}.png'
        mask_path = os.path.join(masks_folder, mask_name)
        mask = cv2.imread(mask_path)
        if mask is None:
            print(f"  WARNING: could not load {mask_path}")
            mask = np.zeros((MASK_H, MASK_W, 3), dtype=np.uint8)
        mask = resize_to(mask, MASK_W, MASK_H)
        label_text = f"{mlabel} | {layer_labels[li]}"
        mask = add_label(mask, label_text, font_scale=0.7, thickness=1)
        row_imgs.append(mask)
    row = np.hstack([row_imgs[0], pad_h, row_imgs[1], pad_h, row_imgs[2]])
    rows.append(row)
 
fig3 = np.vstack([rows[0], pad_v, rows[1], pad_v, rows[2]])
cv2.imwrite(os.path.join(figures_folder, 'fig3_mask_grid.png'), fig3)
print("  Saved fig3_mask_grid.png")
 
# ── FIGURE 4: Side by side layer masks for best model (DAv2) ──────────
print("Making Figure 4: DAv2 layer cutouts on original...")
orig_small = resize_to(orig, TARGET_W, TARGET_H)
orig_labeled2 = add_label(orig_small, "Original")
 
cutout_imgs = [orig_labeled2]
for li in range(3):
    mask_path = os.path.join(masks_folder, f'DepthAnythingV2_layer_{li}.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    mask = resize_to(mask, TARGET_W, TARGET_H)
    # Apply mask to original
    cutout = orig_small.copy()
    cutout[mask < 128] = [30, 30, 30]
    cutout = add_label(cutout, f'Layer {li} ({["BG","MG","FG"][li]})', font_scale=0.9)
    cutout_imgs.append(cutout)
 
pad_h4 = np.ones((TARGET_H, PAD, 3), dtype=np.uint8) * BG
fig4 = np.hstack([cutout_imgs[0], pad_h4,
                  cutout_imgs[1], pad_h4,
                  cutout_imgs[2], pad_h4,
                  cutout_imgs[3]])
cv2.imwrite(os.path.join(figures_folder, 'fig4_layer_cutouts.png'), fig4)
print("  Saved fig4_layer_cutouts.png")
 
print(f"\nAll figures saved to {figures_folder}")