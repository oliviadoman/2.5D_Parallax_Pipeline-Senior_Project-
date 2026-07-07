# Depth-Layered Reprojection with Occlusion Inpainting
### Automated 2.5D Parallax for LED Volume Production

**Olivia Doman** — CS 191W, Stanford University '26  
Faculty Sponsor: Professor Kayvon Fatahalian

---

## Overview

This project investigates whether a simplified layered 2.5D reprojection pipeline can produce convincing parallax from a single image for LED volume production, without per-shot manual intervention.

**Main finding:** Automated 2.5D reprojection produces perceptually convincing parallax for camera translations up to approximately 80px. Beyond this threshold, hard depth-threshold boundaries and inpainting seams become clearly visible.

---

## Pipeline

```
Single RGB Image
      │
      ▼
1. Depth Estimation       (Depth Anything V2 / MiDaS / ZoeDepth — via ComfyUI)
      │
      ▼
2. Layer Segmentation     (n_layer_split.py — N ∈ {3, 4, 6} layers, union mask strategy)
      │
      ▼
3. RGBA Layer Extraction  (rgba_layers.py — original pixels + mask as alpha)
      │
      ▼
4. Occlusion Inpainting   (Stable Diffusion 1.5 / LaMa — via ComfyUI)
      │
      ▼
5. Compositing & Render   (run_pipeline.py — real-time planar layer stack)
      │
      ▼
6. Video Export           (threshold_videos.py — ffmpeg MP4 output)
```

---

## Results

| Model | Edge Quality | Layer Sep. | Runtime (s) |
|---|---|---|---|
| Depth Anything V2 | High, fine branches | Clean boundaries | 3.2 |
| MiDaS | Moderate, blurred | Noisy at edges | 2.1 |
| ZoeDepth | High contrast, coarse | Fragmented midground | 4.8 |

**Camera motion thresholds:**
- ≤ 30px — No visible artifacts, fully production-ready
- 30–80px — Slight edge fringing, acceptable for wide shots
- > 80px — Hard cutout edges visible, unsuitable for production

---

## Repo Structure

```
parallax-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
│
├── pipeline/
│   ├── run_pipeline.py         # Main entry point — renders parallax video for all 3 models
│   ├── threshold_videos.py     # Renders small/medium/large motion magnitude videos
│   ├── n_layer_split.py        # Depth-to-layer segmentation with union mask strategy
│   ├── rgba_layers.py          # Converts masked layers to RGBA cutouts
│   ├── 2D_renderer.py          # Single-model 2D composite renderer
│   ├── inspect_layers.py       # Debug tool — visualises layers on magenta background
│   ├── make_figures.py         # Generates paper figures from pipeline outputs
│   └── sam.py                  # SAM segmentation experiment (exploratory)
│
├── comfyui/
│   └── workflows/              # ComfyUI node workflows for depth + inpainting
│
├── figures/                    # Paper figures
│
└── paper/
    └── CS191W_Doman_Final.pdf
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/parallax-pipeline.git
cd parallax-pipeline
pip install -r requirements.txt
```

### Dependencies
- Python 3.9+
- OpenCV, NumPy
- PyTorch (for SAM)
- ffmpeg (for video export)
- ComfyUI with Depth Anything V2, MiDaS, ZoeDepth nodes
- Stable Diffusion 1.5 or LaMa (via ComfyUI) for inpainting

---

## Usage

### Full pipeline (all 3 depth models)
```bash
python pipeline/run_pipeline.py
```

### Camera motion threshold videos
```bash
python pipeline/threshold_videos.py
# Outputs: threshold_small.mp4 (±25px), threshold_medium.mp4 (±75px), threshold_large.mp4 (±150px)
```

### Generate paper figures
```bash
python pipeline/make_figures.py
```

### Inspect layer outputs
```bash
python pipeline/inspect_layers.py
```

---

## Key Design Decisions

**Union mask inpainting:** The inpainting mask for each layer uses the union of all foreground masks (`n_layer_split.py`), not just the immediate foreground. This eliminates ghosting artifacts where foreground objects remain visible on background layers during camera movement.

**Cinematic camera path:** `run_pipeline.py` uses fractional Brownian motion (summed harmonics) rather than a linear sweep, producing organic camera movement that reads as handheld rather than mechanical.

**Edge feathering:** RGBA alpha edges are Gaussian-blurred (`EDGE_BLUR_PX = 15`) before compositing to soften hard depth-threshold boundaries.

---

## Paper

Full paper: [`paper/CS191W_Doman_Final.pdf`](paper/CS191W_Doman_Final.pdf)

---

## Acknowledgments

This work was completed as part of CS 191W at Stanford University under the supervision of Professor Kayvon Fatahalian.
