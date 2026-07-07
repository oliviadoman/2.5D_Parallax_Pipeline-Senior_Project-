import cv2
import numpy as np
import os

base         = os.path.expanduser('~/Downloads/spr26/191(dep)/images')
rgba_folder  = os.path.join(base, 'rgba_layers')
inp_folder   = os.path.join(base, 'inpainting')
masks_folder = os.path.join(base, 'masks')
out_folder   = os.path.join(base, 'layer_inspect')
os.makedirs(out_folder, exist_ok=True)

# ── DepthAnythingV2 only ───────────────────────────────────────────────
files = {
    'layer_0_rgba': os.path.join(rgba_folder, 'DepthAnythingV2_layer_0_rgba.png'),
    'layer_1_rgba': os.path.join(rgba_folder, 'DepthAnythingV2_layer_1_rgba.png'),
    'layer_2_rgba': os.path.join(rgba_folder, 'DepthAnythingV2_layer_2_rgba.png'),
    'inpaint_0':    os.path.join(inp_folder,  'Anything_0.png'),
    'inpaint_1':    os.path.join(inp_folder,  'Anything_1.png'),
    'mask_0':       os.path.join(masks_folder,'DepthAnythingV2_layer_0.png'),
    'mask_1':       os.path.join(masks_folder,'DepthAnythingV2_layer_1.png'),
    'mask_2':       os.path.join(masks_folder,'DepthAnythingV2_layer_2.png'),
}

# Bright magenta background so transparent areas are obvious
MAGENTA = (255, 0, 255)

for name, path in files.items():
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  MISSING: {path}")
        continue

    if img.ndim == 2:
        # Grayscale mask — save as-is
        cv2.imwrite(os.path.join(out_folder, f'{name}.png'), img)
        opaque = np.sum(img > 128)
        total  = img.size
        print(f"  {name}: mask  — {opaque}/{total} px white ({100*opaque/total:.1f}%)")
        continue

    if img.shape[2] == 3:
        # No alpha — just save
        cv2.imwrite(os.path.join(out_folder, f'{name}.png'), img)
        print(f"  {name}: RGB (no alpha)")
        continue

    # RGBA — composite over magenta so alpha is visible
    h, w = img.shape[:2]
    bg = np.full((h, w, 3), MAGENTA, dtype=np.uint8)
    a  = img[:, :, 3:4] / 255.0
    composite = (img[:, :, :3] * a + bg * (1 - a)).astype(np.uint8)
    cv2.imwrite(os.path.join(out_folder, f'{name}_on_magenta.png'), composite)

    opaque = np.sum(img[:, :, 3] > 128)
    total  = img[:, :, 3].size
    print(f"  {name}: RGBA — {opaque}/{total} opaque px ({100*opaque/total:.1f}% of frame)")

print(f"\nSaved to {out_folder}")
