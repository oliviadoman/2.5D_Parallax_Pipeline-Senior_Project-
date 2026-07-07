import cv2
import numpy as np
import os
import time
import subprocess

# ── PATHS ─────────────────────────────────────────────────────────────
base              = os.path.expanduser('~/Downloads/spr26/191(dep)/images')
inpainting_folder = os.path.join(base, 'inpainting')
rgba_folder       = os.path.join(base, 'rgba_layers')
masks_folder      = os.path.join(base, 'masks')
frames_base       = os.path.join(base, 'threshold_frames')
videos_folder     = os.path.join(base, 'threshold_videos')
os.makedirs(videos_folder, exist_ok=True)

FPS          = 24
N_FRAMES     = 80
EDGE_BLUR_PX = 15

# ── THREE MOTION MAGNITUDES ────────────────────────────────────────────
configs = [
    { 'name': 'small',  'cam_range': 25,  'cam_vertical': 8  },
    { 'name': 'medium', 'cam_range': 75,  'cam_vertical': 20 },
    { 'name': 'large',  'cam_range': 150, 'cam_vertical': 30 },
]

# ── DEPTH ANYTHING V2 ONLY ─────────────────────────────────────────────
inp_prefix  = 'Anything'
rgba_prefix = 'DepthAnythingV2'
mask_prefix = 'DepthAnythingV2'

def cinematic_path(n_frames, amplitude, n_octaves=5, seed=0):
    rng  = np.random.default_rng(seed)
    t    = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    path = np.zeros(n_frames)
    for octave in range(1, n_octaves + 1):
        phase  = rng.uniform(0, 2 * np.pi)
        weight = 1.0 / octave
        path  += weight * np.sin(octave * t + phase)
    peak = np.max(np.abs(path))
    if peak > 0:
        path = path / peak * amplitude
    return path.astype(int).tolist()

def build_clean_background(inp_0, inp_1, mask_layer1):
    h, w = inp_0.shape[:2]
    if inp_1.shape[:2] != (h, w):
        inp_1 = cv2.resize(inp_1, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_layer1, (w, h), interpolation=cv2.INTER_LINEAR)
    mid_mask = mask_resized[:, :, np.newaxis] / 255.0
    return (inp_1.astype(np.float32) * mid_mask +
            inp_0.astype(np.float32) * (1 - mid_mask)).astype(np.uint8)

# ── LOAD LAYERS ONCE ──────────────────────────────────────────────────
print("Loading layers...")

inp_0 = cv2.imread(os.path.join(inpainting_folder, f'{inp_prefix}_0.png'), cv2.IMREAD_COLOR)
inp_1 = cv2.imread(os.path.join(inpainting_folder, f'{inp_prefix}_1.png'), cv2.IMREAD_COLOR)
mask_layer1 = cv2.imread(os.path.join(masks_folder, f'{mask_prefix}_layer_1.png'), cv2.IMREAD_GRAYSCALE)

clean_bg_bgr = build_clean_background(inp_0, inp_1, mask_layer1)
clean_bg     = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2BGRA)
clean_bg[:, :, 3] = 255

bg_h, bg_w = clean_bg.shape[:2]
layers_base = [clean_bg]

for path in [
    os.path.join(rgba_folder, f'{rgba_prefix}_layer_1_rgba.png'),
    os.path.join(rgba_folder, f'{rgba_prefix}_layer_2_rgba.png'),
]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")
    if img.ndim == 2 or img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.shape[:2] != (bg_h, bg_w):
        img = cv2.resize(img, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    if EDGE_BLUR_PX > 0:
        ksize = EDGE_BLUR_PX * 2 + 1
        img[:, :, 3] = cv2.GaussianBlur(img[:, :, 3], (ksize, ksize), 0)
    layers_base.append(img)
    print(f"  Loaded {os.path.basename(path)}")

parallax_factors = [0.0, 0.25, 0.70]
h, w = layers_base[0].shape[:2]

# ── RENDER ONE VIDEO PER MOTION MAGNITUDE ─────────────────────────────
for cfg in configs:
    name       = cfg['name']
    cam_range  = cfg['cam_range']
    cam_vert   = cfg['cam_vertical']

    print(f"\n── {name.upper()} (±{cam_range}px) ──")

    cam_x = cinematic_path(N_FRAMES, cam_range,  seed=1)
    cam_y = cinematic_path(N_FRAMES, cam_vert,   seed=2)

    frames_folder = os.path.join(frames_base, name)
    os.makedirs(frames_folder, exist_ok=True)

    start = time.time()
    for i, (cx, cy) in enumerate(zip(cam_x, cam_y)):
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        for j, (layer, factor) in enumerate(zip(layers_base, parallax_factors)):
            ox, oy = int(cx * factor), int(cy * factor)
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            border = cv2.BORDER_REPLICATE if j == 0 else cv2.BORDER_CONSTANT
            shifted = cv2.warpAffine(layer, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=border,
                                     borderValue=(0, 0, 0, 0))
            alpha = shifted[:, :, 3:4] / 255.0
            canvas[:, :, :3] = (
                shifted[:, :, :3] * alpha +
                canvas[:, :, :3] * (1 - alpha)
            ).astype(np.uint8)
            canvas[:, :, 3] = np.maximum(canvas[:, :, 3], shifted[:, :, 3])

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.join(frames_folder, f'frame_{i:04d}.png'), rgb)

    print(f"  Rendered {N_FRAMES} frames in {time.time()-start:.1f}s")

    output_mp4 = os.path.join(videos_folder, f'threshold_{name}.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-r', str(FPS),
        '-i', os.path.join(frames_folder, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_mp4
    ], check=True)
    print(f"  Saved: threshold_{name}.mp4")

print(f"\n✓ Done — videos saved to {videos_folder}")
print("  threshold_small.mp4   — ±25px  (clean)")
print("  threshold_medium.mp4  — ±55px  (acceptable)")
print("  threshold_large.mp4   — ±100px (objectionable)")