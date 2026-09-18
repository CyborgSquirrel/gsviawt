#!/usr/bin/env python3
"""Download the first N meshes from Objaverse-XL, optionally via TRELLIS-500K.

Two --dataset modes for choosing which objects to download:
  - "trellis-500k" (default): TRELLIS-500K
    (https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K) is a curated,
    aesthetic-score-filtered subset of Objaverse-XL. Its per-source CSVs are
    fetched and cached here (oxl doesn't know about them). Only TRELLIS-500K's
    two Objaverse-XL configs (sketchfab, github) are supported -- its
    ABO/3D-FUTURE/HSSD/Toys4k configs aren't Objaverse-XL sources.
  - "objaverse-xl": the full, unfiltered Objaverse-XL annotations for
    --source, via oxl.get_annotations() (which does its own caching).

Either way, the actual mesh files are fetched through the regular
objaverse-xl API.

Usage:
    pip install objaverse
    python scripts/download_objaverse_sketchfab.py --dataset trellis-500k --source sketchfab --count 500
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
TRELLIS_500K_CSV_BY_SOURCE = {
    "sketchfab": "ObjaverseXL_sketchfab.csv",
    "github": "ObjaverseXL_github.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["trellis-500k", "objaverse-xl"], default="trellis-500k", help="TRELLIS-500K's curated subset, or the full Objaverse-XL annotations")
    parser.add_argument("--source", choices=sorted(TRELLIS_500K_CSV_BY_SOURCE), default="sketchfab", help="which Objaverse-XL source to pull from")
    parser.add_argument("--count", type=int, default=500, help="number of meshes to download")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count(), help="parallel download processes")
    parser.add_argument("--download-dir", default="~/.objaverse", help="objaverse-xl cache dir (also holds the cached TRELLIS-500K CSV)")
    parser.add_argument("--refresh", action="store_true", help="re-download cached annotations/CSV even if a cached copy exists")
    return parser.parse_args()


def get_trellis_500k_csv(source: str, download_dir: Path, refresh: bool) -> pd.DataFrame:
    """Download (or reuse a cached copy of) a TRELLIS-500K Objaverse-XL CSV."""
    csv_name = TRELLIS_500K_CSV_BY_SOURCE[source]
    cache_dir = download_dir / "trellis-500k"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / csv_name

    if refresh or not csv_path.exists():
        url = f"{TRELLIS_500K_BASE_URL}/{csv_name}"
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


def get_objects(dataset: str, source: str, count: int, download_dir: Path, refresh: bool) -> pd.DataFrame:
    """Return the first `count` objects as a fileIdentifier/sha256/source frame."""
    if dataset == "trellis-500k":
        trellis = get_trellis_500k_csv(source, download_dir, refresh)
        subset = trellis.head(count)
        return pd.DataFrame(
            {
                "fileIdentifier": subset["file_identifier"],
                "sha256": subset["sha256"],
                "source": source,
            }
        )

    print(f"Loading full Objaverse-XL {source} annotations into {download_dir}...")
    annotations = oxl.get_annotations(download_dir=str(download_dir), refresh=refresh)
    subset = annotations[annotations["source"] == source].head(count)
    return subset[["fileIdentifier", "sha256", "source"]].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser()

    objects = get_objects(args.dataset, args.source, args.count, download_dir, args.refresh)
    print(f"Downloading {len(objects)} {args.source} meshes ({args.dataset}) into {download_dir}...")

    paths = oxl.download_objects(objects=objects, download_dir=str(download_dir), processes=args.processes)

    print(f"Done. {len(paths)} meshes cached under {download_dir}")


if __name__ == "__main__":
    main()
