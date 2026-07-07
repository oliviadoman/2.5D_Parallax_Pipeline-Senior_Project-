import cv2
import numpy as np
import os
import time

base = os.path.expanduser('~/Downloads/spr26/191(dep)/images')
inpainting_folder = os.path.join(base, 'inpainting')
rgba_folder = os.path.join(base, 'rgba_layers')
frames_folder = os.path.join(base, 'composite_frames')
os.makedirs(frames_folder, exist_ok=True)

# Inpainted background + original midground and foreground cutouts
layer_files = [
    os.path.join(inpainting_folder, 'Anything_0.png'),  # background - drawn first
    os.path.join(rgba_folder, 'DepthAnythingV2_layer_1_rgba.png'),  # midground
    os.path.join(rgba_folder, 'DepthAnythingV2_layer_2_rgba.png'),  # foreground - drawn last
]
parallax_factors = [0.15, 0.5, 1.0]

layers = []
for path in layer_files:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    layers.append(img)
    print(f"Loaded {os.path.basename(path)}: {img.shape}")

h, w = layers[0].shape[:2]
camera_positions = list(range(-100, 101, 5))

print(f"\nRendering {len(camera_positions)} frames...")
start = time.time()

for i, cam_x in enumerate(camera_positions):
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    for layer, factor in zip(layers, parallax_factors):
        offset = int(cam_x * factor)
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        shifted = cv2.warpAffine(layer, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))
        alpha = shifted[:, :, 3:4] / 255.0
        canvas[:, :, :3] = (
            shifted[:, :, :3] * alpha +
            canvas[:, :, :3] * (1 - alpha)
        ).astype(np.uint8)
        canvas[:, :, 3] = np.maximum(canvas[:, :, 3], shifted[:, :, 3])

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(os.path.join(frames_folder, f'frame_{i:04d}.png'), rgb)

elapsed = time.time() - start
print(f"Done — {len(camera_positions)} frames in {elapsed:.1f}s")
print(f"Saved to: {frames_folder}")