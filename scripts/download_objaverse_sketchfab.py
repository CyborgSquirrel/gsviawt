#!/usr/bin/env python3
"""Download the first N Sketchfab meshes from the TRELLIS-500K subset of Objaverse-XL.

TRELLIS-500K (https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K) is a
curated, aesthetic-score-filtered subset of Objaverse-XL. Its per-source CSVs
list which objects made the cut; the actual mesh files are still fetched
through the regular objaverse-xl API. Only the ObjaverseXL_sketchfab.csv
config is supported for now.

Usage:
    pip install objaverse
    python scripts/download_objaverse_sketchfab.py --count 500
"""

import argparse
import multiprocessing
import os
from pathlib import Path

import objaverse.xl as oxl
import pandas as pd
import requests
from tqdm import tqdm

TRELLIS_500K_BASE_URL = "https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K/resolve/main"
TRELLIS_500K_CSV = "ObjaverseXL_sketchfab.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="number of meshes to download")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count(), help="parallel download processes")
    parser.add_argument("--download-dir", default="~/.objaverse", help="objaverse-xl cache dir (also holds the cached TRELLIS-500K CSV)")
    parser.add_argument("--refresh-csv", action="store_true", help="re-download the TRELLIS-500K CSV even if a cached copy exists")
    return parser.parse_args()


def get_trellis_500k_csv(download_dir: Path, refresh: bool) -> pd.DataFrame:
    """Download (or reuse a cached copy of) the TRELLIS-500K sketchfab CSV."""
    cache_dir = download_dir / "trellis-500k"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / TRELLIS_500K_CSV

    if refresh or not csv_path.exists():
        url = f"{TRELLIS_500K_BASE_URL}/{TRELLIS_500K_CSV}"
        print(f"Downloading {url} -> {csv_path}")
        tmp_path = csv_path.with_suffix(".csv.tmp")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        os.rename(tmp_path, csv_path)
    else:
        print(f"Using cached {csv_path}")

    return pd.read_csv(csv_path)


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser()

    trellis = get_trellis_500k_csv(download_dir, args.refresh_csv)
    sketchfab = trellis.head(args.count)
    objects = pd.DataFrame(
        {
            "fileIdentifier": sketchfab["file_identifier"],
            "sha256": sketchfab["sha256"],
            "source": "sketchfab",
        }
    )
    print(f"Downloading {len(objects)} Sketchfab meshes into {download_dir}...")

    paths = oxl.download_objects(objects=objects, download_dir=str(download_dir), processes=args.processes)

    print(f"Done. {len(paths)} meshes cached under {download_dir}")


if __name__ == "__main__":
    main()
