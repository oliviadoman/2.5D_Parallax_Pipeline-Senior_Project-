import cv2
import numpy as np
import os

# ── PATHS ──────────────────────────────────────────────────────────────
original_image_path = '../images/original.png'
masks_folder = '../images/inpainting'
output_folder = '../images/rgba_layers'

os.makedirs(output_folder, exist_ok=True)

# ── LOAD ORIGINAL ──────────────────────────────────────────────────────
original = cv2.imread(original_image_path)
if original is None:
    raise FileNotFoundError(f"Could not load image: {original_image_path}")

orig_h, orig_w = original.shape[:2]
print(f"Original image: {orig_w}x{orig_h}")

# ── PROCESS EACH MASK ──────────────────────────────────────────────────
for mask_file in sorted(os.listdir(masks_folder)):
    if not (mask_file.endswith('.png') or mask_file.endswith('.jpg')):
        continue

    mask_path = os.path.join(masks_folder, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Skipping {mask_file} — could not load")
        continue

    # Resize mask to match original if needed
    if mask.shape[:2] != (orig_h, orig_w):
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Build RGBA — original RGB + mask as alpha
    rgba = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask  # white = visible, black = transparent

    # Save
    name = os.path.splitext(mask_file)[0]
    out_path = os.path.join(output_folder, f'{name}_rgba.png')
    cv2.imwrite(out_path, rgba)
    print(f"Saved {name}_rgba.png")

print("Done — RGBA layers saved to:", output_folder)