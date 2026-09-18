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


def composite_grid(images: np.ndarray, cols: int, pad: int, bg: tuple[int, int, int]) -> np.ndarray:
    """RGBA (N,H,W,4) uint8 -> RGB (H',W',3) uint8 contact sheet, alpha
    flattened onto `bg` (transparent film renders would otherwise show as
    black squares)."""
    n, h, w, _ = images.shape
    rows = -(-n // cols)  # ceil div
    bg_arr = np.array(bg, np.uint8)

    grid = np.tile(bg_arr, (rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 1))
    for i in range(n):
        r, c = divmod(i, cols)
        y0 = pad + r * (h + pad)
        x0 = pad + c * (w + pad)
        rgb = images[i, ..., :3].astype(np.float32)
        a = images[i, ..., 3:4].astype(np.float32) / 255.0
        composited = rgb * a + np.array(bg, np.float32) * (1 - a)
        grid[y0:y0 + h, x0:x0 + w] = np.clip(composited + 0.5, 0, 255).astype(np.uint8)
    return grid


def main() -> None:
    args = parse_args()
    with h5py.File(args.h5, "r") as hf:
        images = hf["images"][:]  # N x H x W x 4 uint8, RGBA

    grid = composite_grid(images, args.cols, args.pad, tuple(args.bg))
    rows = -(-images.shape[0] // args.cols)
    Image.fromarray(grid, mode="RGB").save(args.out)
    print(f"wrote {args.out} ({rows}x{args.cols} grid, {images.shape[0]} meshes, {images.shape[2]}x{images.shape[1]} each)")


if __name__ == "__main__":
    main()
