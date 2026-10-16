"""
2.5D parallax layering: v1 (current repo method) and v2 (improved edges + depth relief).

Both variants take the same inputs (RGB image, depth map where near = 1.0) and
produce a list of layers ordered far -> near. Each layer has:
    color  (H, W, 3) float32 in [0, 1], straight (not premultiplied)
    alpha  (H, W)    float32 in [0, 1]
    depth  (H, W)    float32, the depth used to move that layer's pixels

v1 matches the repo: equal depth thresholds, hard masks, a fixed Gaussian blur on
alpha, original pixels as layer color, the far plate inpainted under the union of
all foreground masks, and each layer moved as one flat card.

v2 changes five things:
  1. cut points at gaps in the depth histogram (multi-level Otsu)
  2. alpha refined by an image-guided filter so soft edges follow real image edges
  3. color decontamination plus edge extension, so edge pixels stop carrying
     background color along with the foreground
  4. the far plate is only filled where the camera can actually reveal it
  5. each layer keeps its own depth relief instead of being a flat card
"""
import cv2
import numpy as np
from skimage.filters import threshold_multiotsu


# ---------- helpers ----------

def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r, eps):
    """He et al. guided filter. guide: (H,W) gray in [0,1], src: (H,W)."""
    mI, mp = box(guide, r), box(src, r)
    cov = box(guide * src, r) - mI * mp
    var = box(guide * guide, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a, r) * guide + box(b, r)


def inpaint(img01, hole, radius=5):
    """OpenCV Telea fill. Stand-in for LaMa / SD in the real pipeline."""
    if not hole.any():
        return img01.copy()
    u8 = np.clip(img01 * 255 + 0.5, 0, 255).astype(np.uint8)
    out = cv2.inpaint(u8, hole.astype(np.uint8), radius, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255


def pushpull(color, known):
    """Fill unknown pixels from known ones only (pyramid push-pull).
    Unlike Telea on a zero-padded image, nothing outside `known` leaks in."""
    w = known.astype(np.float32)
    levels_c, levels_w = [color * w[..., None]], [w]
    while min(levels_w[-1].shape) > 4:
        levels_c.append(cv2.pyrDown(levels_c[-1]))
        levels_w.append(cv2.pyrDown(levels_w[-1]))
    c = levels_c[-1] / np.maximum(levels_w[-1][..., None], 1e-6)
    for lc, lw in zip(reversed(levels_c[:-1]), reversed(levels_w[:-1])):
        up = cv2.pyrUp(c, dstsize=(lw.shape[1], lw.shape[0]))
        here = lc / np.maximum(lw[..., None], 1e-6)
        t = np.clip(lw * 4, 0, 1)[..., None]
        c = t * here + (1 - t) * up
    return np.where(known[..., None], color, c).astype(np.float32)


def disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def extend_depth(depth, known, sigma=12.0, lo=None, hi=None):
    """Push a layer's depth outward from its own pixels, smooth it, and clamp it
    to the layer's own depth range so thin structures (branches against sky,
    where the depth map is blurry) are not stretched like a rubber sheet."""
    known = cv2.erode(known.astype(np.uint8), disk(2)) > 0 if known.sum() > 5000 else known
    if lo is not None:
        depth = np.clip(depth, lo, hi)
    num = cv2.GaussianBlur(depth * known, (0, 0), 25)
    den = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 25)
    filled = np.where(known, depth, num / np.maximum(den, 1e-4))
    out = cv2.GaussianBlur(filled, (0, 0), sigma)
    return out if lo is None else np.clip(out, lo, hi)


# ---------- v1: current repo method ----------

def build_v1(img, depth, n_layers=3, edge_blur_px=15):
    lo, hi = float(depth.min()), float(depth.max())
    cuts = [lo + (hi - lo) * k / n_layers for k in range(1, n_layers)]
    labels = np.digitize(depth, cuts)                      # 0 = far
    masks = [(labels == k) for k in range(n_layers)]

    layers = []
    for k in range(n_layers):
        in_front = np.zeros_like(masks[0])
        for j in range(k + 1, n_layers):
            in_front |= masks[j]
        if k == 0:
            color = inpaint(img, in_front)                 # full union fill
            alpha = np.ones(depth.shape, np.float32)
        else:
            color = img.copy()                             # original pixels, no cleanup
            sig = edge_blur_px / 3.0
            alpha = cv2.GaussianBlur(masks[k].astype(np.float32), (0, 0), sig)
        flat = float(depth[masks[k]].mean())
        layers.append(dict(color=color, alpha=alpha,
                           depth=np.full(depth.shape, flat, np.float32)))
    return layers, cuts


# ---------- v2 ----------

def build_v2(img, depth, n_layers=3, max_move=160, gf_radius=6, gf_eps=2e-3, relief=True):
    cuts = list(threshold_multiotsu(depth, classes=n_layers))
    labels = np.digitize(depth, cuts)
    masks = [(labels == k) for k in range(n_layers)]
    edges = [0.0] + [float(c) for c in cuts] + [1.0]
    bounds = [(edges[k], edges[k + 1]) for k in range(n_layers)]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. soft, image-guided alpha for each non-far layer.
    #    "Near side" = this layer or anything in front of it, so alphas nest
    #    and adjacent layers never leave a gap between them.
    alphas = [np.ones(depth.shape, np.float32)]
    for k in range(1, n_layers):
        near_side = (labels >= k).astype(np.float32)
        a = np.clip(guided_filter(gray, near_side, gf_radius, gf_eps), 0, 1)
        band = cv2.dilate(near_side, disk(gf_radius * 2)) > 0
        a = np.where(band, a, 0).astype(np.float32)
        # nested: this layer stays opaque under nearer layers (filled below),
        # so nothing behind it shows through at their shared edge
        alphas.append(a)

    # 4. far plate: fill only what a move of max_move px can uncover
    occluded = labels >= 1
    span = float(depth[occluded].max() - depth[labels == 0].mean()) if occluded.any() else 0
    reveal = int(np.ceil(max_move * span)) + 8
    visible = (~occluded).astype(np.uint8)
    reach = cv2.dilate(visible, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * reveal + 1, 2 * reveal // 3 + 1))) > 0
    hole = occluded & reach
    plate = inpaint(img, hole)

    d0 = extend_depth(depth, ~occluded, lo=bounds[0][0], hi=bounds[0][1]) if relief else np.full(depth.shape, float(depth[~occluded].mean()), np.float32)
    layers = [dict(color=plate, alpha=alphas[0], depth=d0)]

    behind = plate
    for k in range(1, n_layers):
        a = alphas[k]
        own = labels == k
        nearer = labels > k
        a3 = a[..., None]
        # 3a. decontaminate the soft edge facing farther layers:
        #     I = a*F + (1-a)*B  ->  F = (I - (1-a)B) / a
        #     B is estimated from real background pixels right next to the edge
        #     (alpha ~ 0), not from the inpainted plate, which can be off in tone
        B = pushpull(img, a < 0.03)
        F = np.where(a3 > 0.25, (img - (1 - a3) * B) / np.maximum(a3, 1e-3), img)
        F = np.clip(F, 0, 1).astype(np.float32)
        # 3b. trusted pixels: solid alpha, own label, a few px away from nearer layers
        known = (a > 0.97) & own & ~(cv2.dilate(nearer.astype(np.uint8), disk(3)) > 0)
        # grow color outward past soft edges, and under nearer layers only as far
        # as the camera can reveal (same idea as the far plate)
        if nearer.any():
            span_k = float(depth[nearer].max() - depth[own].mean())
            rk = int(np.ceil(max_move * span_k)) + 8
        else:
            rk = 0
        reach_k = cv2.dilate(known.astype(np.uint8), disk(max(rk, gf_radius * 3))) > 0
        unknown = (a > 0.003) & ~known & reach_k
        fill = pushpull(F, known)
        w = np.clip((a - 0.25) / 0.5, 0, 1)[..., None] * ~nearer[..., None]
        color = np.where(unknown[..., None], w * F + (1 - w) * fill, F).astype(np.float32)
        a = np.where(nearer & ~reach_k, 0, a).astype(np.float32)   # never revealed
        a3 = a[..., None]

        # 5. depth relief: this layer's own depth, extended past its edges
        d_k = extend_depth(depth, own, lo=bounds[k][0], hi=bounds[k][1]) if relief else np.full(depth.shape, float(depth[own].mean()), np.float32)
        layers.append(dict(color=color, alpha=a, depth=d_k))
        behind = a3 * color + (1 - a3) * behind
    return layers, cuts


# ---------- rendering ----------

def render(layers, dx, dy=0.0):
    """Composite far -> near. Each pixel moves by (dx, dy) * its layer depth."""
    H, W = layers[0]['alpha'].shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = np.zeros((H, W, 3), np.float32)
    dx, dy = np.float32(dx), np.float32(dy)
    for L in layers:
        d = L['depth']
        # backward map, two fixed-point steps so relief layers sample correctly
        mx, my = xs - dx * d, ys - dy * d
        for _ in range(2):
            dd = cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mx, my = xs - dx * dd, ys - dy * dd
        pm = L['color'] * L['alpha'][..., None]            # premultiplied
        c = cv2.remap(pm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.remap(L['alpha'], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., None]
        out = c + (1 - a) * out
    return np.clip(out, 0, 1)


def load(image_path, depth_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = (d - d.min()) / max(1e-6, d.max() - d.min())      # near = 1
    return img, d
