"""
Edge-artifact metric for layered 2.5D parallax.

There is no ground-truth photo from a moved camera, so the reference is a
per-pixel depth warp: every pixel of the original moves by exactly its own depth,
nearest pixel wins. Pixels this reference cannot see (disocclusions) are skipped,
because any fill there is a guess. Everywhere else, a perfect layered render would
match the reference, so the difference is error introduced by layering: flat-card
distortion, halos, fringes and seams.

The score is the mean absolute RGB error (0-255 scale) inside a band around
depth edges in the moved view, which is where layering artifacts live.
"""
import cv2
import numpy as np


def reference_view(img, depth, dx, dy=0.0):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    tx = np.rint(xs + dx * depth).astype(int)
    ty = np.rint(ys + dy * depth).astype(int)
    ok = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
    order = np.argsort(depth[ok], kind='stable')            # far first, near overwrites
    sx, sy = tx[ok][order], ty[ok][order]
    out = np.zeros_like(img)
    zbuf = np.full((H, W), -1.0, np.float32)
    out[sy, sx] = img[ys[ok][order], xs[ok][order]]
    zbuf[sy, sx] = depth[ok][order]
    valid = zbuf >= 0
    # drop 1px cracks from rounding: require a solid neighborhood
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return out, zbuf, valid


def edge_band(zbuf, valid, width=8, jump=0.04):
    z = np.where(valid, zbuf, 0).astype(np.float32)
    gx = np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)) / 4
    gy = np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)) / 4
    edges = ((gx + gy) > jump) & valid
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return (cv2.dilate(edges.astype(np.uint8), k) > 0) & valid


def score(render, img, depth, dx, dy=0.0, crop=None):
    ref, zbuf, valid = reference_view(img, depth, dx, dy)
    band = edge_band(zbuf, valid)
    if crop is not None:                                      # ignore frame borders
        m = np.zeros_like(band)
        m[crop:-crop, crop:-crop] = True
        band &= m
        valid &= m
    err = np.abs(render - ref).mean(axis=2) * 255
    return dict(edge=float(err[band].mean()), frame=float(err[valid].mean()),
                band_px=int(band.sum()))
