#!/usr/bin/env python3
"""Turn a render_objaverse.py turntable HDF5 (multiple views per mesh,
recorded via `mesh_index`) into a single rows x cols grid video -- every
cell spins in sync through the same turntable, one frame of video per
azimuth step. Meshes that failed to render (see `failed_mesh_indices`
attr) get a blank cell.

Usage:
    python scripts/make_grid_video.py --h5 bla/trellis25_tt.h5 --out bla/trellis25_grid.mp4 --cols 5 --fps 15
"""

import argparse
import json

import h5py
import numpy as np

from make_grid_png import composite_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", required=True, help="render_objaverse.py turntable output")
    parser.add_argument("--out", required=True, help="output .mp4 path")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better/bigger)")
    parser.add_argument("--pad", type=int, default=4, help="pixels of padding between cells")
    parser.add_argument("--bg", type=int, nargs=3, default=(32, 32, 32), help="RGB padding/background color")
    return parser.parse_args()


def group_by_mesh(images: np.ndarray, mesh_index: np.ndarray, n_meshes: int) -> list[np.ndarray]:
    """Per-mesh view stacks, in mesh order, each (num_frames, H, W, 4).
    Views for a mesh appear consecutively in render order, so this is just
    a split -- but grouping explicitly by index is robust to failed meshes
    contributing zero rows."""
    groups: list[list[np.ndarray]] = [[] for _ in range(n_meshes)]
    for img, mi in zip(images, mesh_index):
        groups[int(mi)].append(img)
    return [np.stack(g, axis=0) if g else np.zeros((0,), np.uint8) for g in groups]


def main() -> None:
    args = parse_args()
    import imageio.v2 as imageio

    with h5py.File(args.h5, "r") as hf:
        images = hf["images"][:]
        mesh_index = hf["mesh_index"][:]
        n_meshes = hf["mesh_paths"].shape[0]
        failed = set(json.loads(hf.attrs["failed_mesh_indices"])) if "failed_mesh_indices" in hf.attrs else set()

    h, w = images.shape[1:3]
    per_mesh = group_by_mesh(images, mesh_index, n_meshes)
    num_frames = max((g.shape[0] for g in per_mesh), default=0)
    if num_frames == 0:
        raise SystemExit("no views found in h5 (every mesh failed?)")

    blank = np.zeros((h, w, 4), np.uint8)
    for i, g in enumerate(per_mesh):
        if g.shape[0] == 0:
            print(f"mesh {i} failed to render -- using a blank cell")
        elif g.shape[0] != num_frames:
            print(f"mesh {i} has {g.shape[0]}/{num_frames} frames -- holding its last frame")

    writer = imageio.get_writer(
        args.out, format="FFMPEG", mode="I", fps=args.fps,
        codec="libx264", macro_block_size=1,
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", str(args.crf), "-preset", "medium"],
    )
    try:
        for t in range(num_frames):
            frame_images = np.stack([
                g[min(t, g.shape[0] - 1)] if g.shape[0] > 0 else blank
                for g in per_mesh
            ], axis=0)
            grid = composite_grid(frame_images, args.cols, args.pad, tuple(args.bg))
            writer.append_data(grid)
    finally:
        writer.close()

    rows = -(-n_meshes // args.cols)
    print(f"wrote {args.out} ({rows}x{args.cols} grid, {n_meshes} meshes, {num_frames} frames @ {args.fps}fps, "
          f"{len(failed)} failed)")


if __name__ == "__main__":
    main()
