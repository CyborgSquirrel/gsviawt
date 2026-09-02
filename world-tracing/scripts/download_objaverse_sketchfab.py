#!/usr/bin/env python3
"""Download the first N Sketchfab meshes from Objaverse-XL.

Usage:
    pip install objaverse
    python scripts/download_objaverse_sketchfab.py --count 500
"""

import argparse
import multiprocessing

import objaverse.xl as oxl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="number of meshes to download")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count(), help="parallel download processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotations = oxl.get_annotations()
    sketchfab = annotations[annotations["source"] == "sketchfab"].head(args.count)
    print(f"Downloading {len(sketchfab)} Sketchfab meshes into ~/.objaverse...")

    paths = oxl.download_objects(objects=sketchfab, processes=args.processes)

    print(f"Done. {len(paths)} meshes cached under ~/.objaverse")


if __name__ == "__main__":
    main()
