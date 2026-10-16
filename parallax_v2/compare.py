"""
Compare v1 (repo method) and v2 on one image.

    python compare.py original.png depth.png out_dir [--move 120]

Writes:
    edge_error.csv / edge_error.png   score vs camera move for v1, v2 (flat), v2
    compare_<move>px.mp4              v1 on top, v2 below, same camera path
    still_<move>px.png                one frame at the far end of the move
"""
import argparse
import csv
import os
import subprocess

import cv2
import numpy as np

from layering import load, build_v1, build_v2, render
from edge_metric import score


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def label(frame, text):
    cv2.rectangle(frame, (0, 0), (len(text) * 11 + 24, 34), (22, 23, 15), -1)
    cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (216, 230, 236), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image'), ap.add_argument('depth'), ap.add_argument('out')
    ap.add_argument('--move', type=float, default=120)
    ap.add_argument('--seconds', type=float, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img, depth = load(args.image, args.depth)
    variants = {
        'v1 (current)': build_v1(img, depth)[0],
        'v2 edges only (flat cards)': build_v2(img, depth, relief=False)[0],
        'v2 (edges + depth relief)': build_v2(img, depth)[0],
    }

    # ---- metric sweep ----
    moves = list(range(0, 161, 10))
    crop = 170                                       # ignore stretched frame borders
    rows = []
    for m in moves:
        row = {'move_px': m}
        for name, L in variants.items():
            row[name] = round(score(render(L, m), img, depth, m, crop=crop)['edge'], 3)
        rows.append(row)
    with open(os.path.join(args.out, 'edge_error.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)

    # ---- video: same camera path for every variant ----
    H, W = depth.shape
    m = args.move
    pad_x, pad_y = int(m) + 10, int(m * 0.18) + 6
    fps, n = 30, int(args.seconds * 30)
    tmp = os.path.join(args.out, '_frames')
    os.makedirs(tmp, exist_ok=True)
    show = ['v1 (current)', 'v2 (edges + depth relief)']
    for i in range(n):
        t = i / n * 2 * np.pi
        dx = m * (0.8 * np.sin(t) + 0.2 * np.sin(2 * t + 0.7))     # handheld-ish sweep
        dy = m * 0.15 * np.sin(t * 2 + 1.3)
        tiles = []
        for name in show:
            fr = to_u8(render(variants[name], dx, dy))[pad_y:H - pad_y, pad_x:W - pad_x]
            fr = cv2.resize(fr, (960, int(960 * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA)
            tiles.append(label(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), f'{name}   move {int(m)}px'))
        gap = np.full((6, 960, 3), (15, 23, 22), np.uint8)
        frame = np.vstack([tiles[0], gap, tiles[1]])
        if frame.shape[0] % 2:
            frame = frame[:-1]
        cv2.imwrite(os.path.join(tmp, f'{i:04d}.png'), frame)
        if i == n // 4:
            cv2.imwrite(os.path.join(args.out, f'still_{int(m)}px.png'), frame)
    mp4 = os.path.join(args.out, f'compare_{int(m)}px.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i',
                    os.path.join(tmp, '%04d.png'), '-c:v', 'libx264', '-crf', '22',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(open(os.path.join(args.out, 'edge_error.csv')).read())


if __name__ == '__main__':
    main()
