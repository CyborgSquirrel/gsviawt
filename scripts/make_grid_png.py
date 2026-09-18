#!/usr/bin/env python3
"""Tile an "images" dataset from a render_objaverse.py HDF5 output into a
single rows x cols contact-sheet PNG (one view per mesh -- pair with
view_strategy=front so each mesh contributes exactly one image).

Usage:
    python scripts/make_grid_png.py --h5 bla/trellis25.h5 --out bla/trellis25_grid.png --cols 5
"""

import argparse

import h5py
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", required=True, help="render_objaverse.py output with an 'images' dataset")
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--pad", type=int, default=4, help="pixels of padding between cells")
    parser.add_argument("--bg", type=int, nargs=3, default=(32, 32, 32), help="RGB padding/background color")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.h5, "r") as hf:
        images = hf["images"][:]  # N x H x W x 4 uint8, RGBA

    n, h, w, _ = images.shape
    cols = args.cols
    rows = -(-n // cols)  # ceil div

    bg = np.array(list(args.bg) + [255], dtype=np.uint8)
    grid = np.tile(bg, (rows * h + (rows + 1) * args.pad, cols * w + (cols + 1) * args.pad, 1))

    for i in range(n):
        r, c = divmod(i, cols)
        y0 = args.pad + r * (h + args.pad)
        x0 = args.pad + c * (w + args.pad)
        # flatten alpha onto the background color so transparent film renders
        # don't just show as black squares in the grid
        rgb = images[i, ..., :3].astype(np.float32)
        a = images[i, ..., 3:4].astype(np.float32) / 255.0
        composited = rgb * a + np.array(args.bg, np.float32) * (1 - a)
        grid[y0:y0 + h, x0:x0 + w, :3] = np.clip(composited + 0.5, 0, 255).astype(np.uint8)
        grid[y0:y0 + h, x0:x0 + w, 3] = 255

    Image.fromarray(grid, mode="RGBA").convert("RGB").save(args.out)
    print(f"wrote {args.out} ({rows}x{cols} grid, {n} meshes, {w}x{h} each)")


if __name__ == "__main__":
    main()
