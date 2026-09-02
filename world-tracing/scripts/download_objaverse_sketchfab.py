#!/usr/bin/env python3
"""Download the first N meshes from Objaverse 1.0 (all sourced from Sketchfab).

Usage:
    pip install objaverse
    python scripts/download_objaverse_sketchfab.py --count 500 --out-dir ./objaverse_meshes
"""

import argparse
import multiprocessing
import shutil
from pathlib import Path

import objaverse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="number of meshes to download")
    parser.add_argument("--out-dir", type=Path, default=Path("objaverse_meshes"), help="directory to copy the downloaded .glb files into")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count(), help="parallel download processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    uids = objaverse.load_uids()[: args.count]
    print(f"Downloading {len(uids)} meshes from Objaverse (Sketchfab)...")

    objects = objaverse.load_objects(uids=uids, download_processes=args.processes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for uid, src_path in objects.items():
        dst_path = args.out_dir / f"{uid}.glb"
        shutil.copy(src_path, dst_path)

    print(f"Done. {len(objects)} meshes copied to {args.out_dir}")


if __name__ == "__main__":
    main()
