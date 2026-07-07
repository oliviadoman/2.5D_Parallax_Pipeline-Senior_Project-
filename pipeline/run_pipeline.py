import cv2
import numpy as np
import os
import time
import subprocess

# ── PATHS ─────────────────────────────────────────────────────────────
base              = os.path.expanduser('~/Downloads/spr26/191(dep)/images')
inpainting_folder = os.path.join(base, 'inpainting copy')
rgba_folder       = os.path.join(base, 'rgba_layers')
masks_folder      = os.path.join(base, 'masks')
frames_base       = os.path.join(base, 'composite_frames')
videos_folder     = base

# Background(static) → Midground(slow) → Foreground(fast)
parallax_factors  = [0.1, 0.35, 0.75]
FPS               = 24

# ── TUNING ────────────────────────────────────────────────────────────
N_FRAMES      = 120   # frames — 5 s at 24 fps; longer = slower, more cinematic
CAM_RANGE     = 80    # max horizontal pixel shift
CAM_VERTICAL  = 25    # max vertical pixel shift
EDGE_BLUR_PX  = 15    # feather radius on RGBA alpha edges; 0 = off


def cinematic_path(n_frames, amplitude, n_octaves=5, seed=0):
    """
    Organic camera path: sums several harmonics at integer multiples of the
    base frequency so the loop closes perfectly, with each higher octave
    contributing less (like fractional Brownian motion).  The result is a
    smooth, non-repeating-within-the-loop drift that reads as handheld/alive
    rather than mechanical.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    path = np.zeros(n_frames)
    for octave in range(1, n_octaves + 1):
        phase  = rng.uniform(0, 2 * np.pi)
        weight = 1.0 / octave          # lower octaves dominate — slow drift wins
        path  += weight * np.sin(octave * t + phase)
    # Normalise so the peak actually reaches the requested amplitude
    peak = np.max(np.abs(path))
    if peak > 0:
        path = path / peak * amplitude
    return path.astype(int).tolist()


cam_x_positions = cinematic_path(N_FRAMES, CAM_RANGE,    seed=1)
cam_y_positions = cinematic_path(N_FRAMES, CAM_VERTICAL, seed=2)

# ── MODEL CONFIG ──────────────────────────────────────────────────────
# (model_name, inpainting prefix, rgba prefix, mask prefix)
models = [
    ('DepthAnythingV2', 'Anything',  'DepthAnythingV2', 'DepthAnythingV2'),
    ('MiDaSDepthMap',   'Midas',     'MiDaSDepthMap',   'MiDaSDepthMap'),
    ('ZoeDepthMap',     'Zoe',       'ZoeDepthMap',     'ZoeDepthMap'),
]


def build_clean_background(inp_0, inp_1, mask_layer1):
    """
    Synthesise a background with BOTH foreground and midground removed.

    inp_0      : Anything_0.png — foreground (layer_0) inpainted away, midground still visible
    inp_1      : Anything_1.png — midground  (layer_1) inpainted away, foreground still visible
    mask_layer1: binary mask, white = midground pixels

    Strategy:
      - Where the midground mask is white → use inp_1 pixels (midground already gone there)
      - Everywhere else                  → use inp_0 pixels (foreground already gone there)
    Result: neither foreground nor midground content remains.
    """
    h, w = inp_0.shape[:2]
    if inp_1.shape[:2] != (h, w):
        inp_1 = cv2.resize(inp_1, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_layer1, (w, h), interpolation=cv2.INTER_LINEAR)
    mid_mask = mask_resized[:, :, np.newaxis] / 255.0   # (H, W, 1) in [0,1]
    bg = (inp_1.astype(np.float32) * mid_mask + inp_0.astype(np.float32) * (1 - mid_mask)).astype(np.uint8)
    return bg


for model_name, inp_prefix, rgba_prefix, mask_prefix in models:
    print(f"\n{'='*50}")
    print(f"  Model: {model_name}")
    print(f"{'='*50}")

    frames_folder = os.path.join(frames_base, model_name)
    os.makedirs(frames_folder, exist_ok=True)

    # ── Load inpaintings ───────────────────────────────────────────────
    inp_0 = cv2.imread(os.path.join(inpainting_folder, f'{inp_prefix}_0.png'), cv2.IMREAD_COLOR)
    inp_1 = cv2.imread(os.path.join(inpainting_folder, f'{inp_prefix}_1.png'), cv2.IMREAD_COLOR)
    if inp_0 is None or inp_1 is None:
        raise FileNotFoundError(f"Missing inpainting files for {model_name}")

    # ── Load midground mask (grayscale) ───────────────────────────────
    mask_path = os.path.join(masks_folder, f'{mask_prefix}_layer_1.png')
    mask_layer1 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_layer1 is None:
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    # ── Build clean background ─────────────────────────────────────────
    clean_bg_bgr = build_clean_background(inp_0, inp_1, mask_layer1)
    clean_bg = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2BGRA)
    clean_bg[:, :, 3] = 255  # fully opaque — always fills the canvas
    print(f"  Built clean background (foreground + midground removed)")

    # ── Load RGBA cutout layers ────────────────────────────────────────
    # layer_0 = background scene (sky, trees, road) → slow, drawn behind
    # layer_1 = close tree trunks on sides        → fast, drawn in front
    layer0_path = os.path.join(rgba_folder, f'{rgba_prefix}_layer_0_rgba.png')
    layer1_path = os.path.join(rgba_folder, f'{rgba_prefix}_layer_1_rgba.png')

    bg_h, bg_w = clean_bg.shape[:2]
    layers = [clean_bg]
    for path in [layer0_path, layer1_path]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load {path}")
        if img.ndim == 2 or img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        if img.shape[:2] != (bg_h, bg_w):
            img = cv2.resize(img, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
        if EDGE_BLUR_PX > 0:
            ksize = EDGE_BLUR_PX * 2 + 1  # GaussianBlur requires odd kernel size
            img[:, :, 3] = cv2.GaussianBlur(img[:, :, 3], (ksize, ksize), 0)
        layers.append(img)
        print(f"  Loaded {os.path.basename(path)}: {img.shape}")

    h, w = layers[0].shape[:2]

    print(f"\n  Rendering {len(cam_x_positions)} frames...")
    start = time.time()

    for i, (cam_x, cam_y) in enumerate(zip(cam_x_positions, cam_y_positions)):
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        for j, (layer, factor) in enumerate(zip(layers, parallax_factors)):
            offset_x = int(cam_x * factor)
            offset_y = int(cam_y * factor)
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            # Background stretches its edge pixels so no black gap appears;
            # RGBA cutout layers go transparent at the edge so the background shows through.
            border_mode = cv2.BORDER_REPLICATE if j == 0 else cv2.BORDER_CONSTANT
            shifted = cv2.warpAffine(layer, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=border_mode,
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
    print(f"  Done — {len(cam_x_positions)} frames in {elapsed:.1f}s")

    output_mp4 = os.path.join(videos_folder, f'parallax_{model_name}.mp4')
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-r', str(FPS),
        '-i', os.path.join(frames_folder, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_mp4
    ]
    print(f"\n  Encoding → {os.path.basename(output_mp4)}")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"  Saved: {output_mp4}")

print(f"\n✓ All done — 3 videos saved to {videos_folder}")
