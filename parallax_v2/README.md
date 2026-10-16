# Parallax v2 prototype

A drop-in experiment for the 2.5D Parallax Pipeline. It compares the current
layering method (v1) against an improved one (v2) on the same image and depth map,
and scores both with an edge-artifact metric across camera moves.

## Run

```bash
pip install opencv-python numpy scikit-image
python compare.py original.png "depth maps/DepthAnythingV2.png" out --move 120
```

Outputs in `out/`: `edge_error.csv` (score vs camera move), `compare_120px.mp4`
(v1 on top, v2 below, same camera path), and `still_120px.png`.

## What v2 changes

| | v1 (current repo) | v2 |
|---|---|---|
| Cut points | Equal depth thresholds | Gaps in the depth histogram (multi-level Otsu) |
| Alpha | Hard mask + 15px Gaussian blur | Image-guided filter, so soft edges follow real edges; layers nest so no gaps |
| Edge color | Original pixels (carry background color) | Decontaminated using a local background estimate, then extended outward |
| Far plate fill | Union of all foreground masks | Only the area a move of `max_move` px can reveal |
| Geometry | Each layer is a flat card | Each layer keeps its own smoothed depth relief, clamped to its depth range |

## The metric (`edge_metric.py`)

There is no real photo from a moved camera, so the reference is a per-pixel depth
warp of the original: every pixel moves by exactly its own depth, nearest wins.
Pixels the reference cannot see (disocclusions) are skipped, since any fill there
is a guess. The score is mean absolute RGB error (0 to 255) in an 8px band around
depth edges, ignoring a 170px frame border.

Calibration: the README's visual "clean" limit for v1 is 30px, where v1 scores
about 20. Any move that keeps a variant under 20 is treated as equally clean.

## Results on the forest image (3 layers, Depth Anything V2)

| Move | v1 | v2 edges only | v2 full |
|---|---|---|---|
| 0px | 0.8 | 2.7 | 2.7 |
| 30px | 20.0 | 19.6 | 8.9 |
| 80px | 32.0 | 30.6 | 13.6 |
| 120px | 36.4 | 34.6 | 16.8 |
| 160px | 39.3 | 38.1 | 19.7 |

v2 stays under v1's 30px error level all the way to 160px, the largest move tested.

## Honest notes

- Almost all of the measured gain comes from depth relief. The edge cleanup on its
  own barely moves the score, and the visible ghosts in both versions come from
  the disocclusion fill, not the alpha edges.
- The metric uses the depth map as truth, so it rewards following it. Where the
  depth map is wrong (thin branches bleeding into sky) following it bends the
  branches. `extend_depth(sigma=...)` trades that off: higher sigma means stiffer,
  less bending, slightly higher score. 12 is the default.
- Fills here use OpenCV Telea as a stand-in so both versions are equal. Swap
  `inpaint()` for the LaMa or SD 1.5 ComfyUI call from the main pipeline; that is
  the next biggest visual improvement.
- v2 at rest (0px) is very slightly softer than the original (score 2.7) because
  of the edge decontamination. Not visible at normal size.
