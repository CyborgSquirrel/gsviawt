#!/usr/bin/env python3
"""Download the first N Sketchfab meshes from Objaverse-XL, optionally via TRELLIS-500K.

Two --dataset modes for choosing which objects to download:
  - "trellis-500k" (default): TRELLIS-500K
    (https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K) is a curated,
    aesthetic-score-filtered subset of Objaverse-XL. Its sketchfab CSV is
    fetched and cached here (oxl doesn't know about it).
  - "objaverse-xl": the full, unfiltered Objaverse-XL sketchfab annotations,
    via oxl.get_annotations() (which does its own caching).

Either way, the actual mesh files are fetched through the regular
objaverse-xl API.

By default the first --count objects are taken. With --sample-seed they are a seeded random
sample of the whole list instead, and with --have-file, GLBs that are already downloaded (one
path per line, the file stem is the object's uid) count towards --count and are not downloaded
again: --count is then the total, and only the missing ones are fetched.

Usage:
    pip install objaverse
    python scripts/download_objaverse_sketchfab.py --dataset trellis-500k --count 500
    python scripts/download_objaverse_sketchfab.py --count 10000 --sample-seed 0 --have-file bla/trellis_glbs.txt --download-dir /mnt/hdd/objaverse
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["trellis-500k", "objaverse-xl"], default="trellis-500k", help="TRELLIS-500K's curated subset, or the full Objaverse-XL annotations")
    parser.add_argument("--count", type=int, default=500, help="number of meshes to download (with --have-file: the total, counting the ones already downloaded)")
    parser.add_argument("--sample-seed", type=int, default=None, help="take a random sample of --count objects with this seed, instead of the first --count")
    parser.add_argument("--have-file", default=None, help="text file of already downloaded GLB paths: they count towards --count and are not downloaded again")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count(), help="parallel download processes")
    parser.add_argument("--download-dir", default="~/.objaverse", help="objaverse-xl cache dir (also holds the cached TRELLIS-500K CSV)")
    parser.add_argument("--refresh", action="store_true", help="re-download cached annotations/CSV even if a cached copy exists")
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


def pick(frame: pd.DataFrame, count: int, sample_seed: int | None, have_uids: set[str], uid_column: str) -> pd.DataFrame:
    """`count` rows in total, the already downloaded ones (have_uids) counting towards it: the
    first rows or a seeded random sample of the rest, never one of have_uids."""
    need = max(count - len(have_uids), 0)
    rest = frame[~frame[uid_column].str.rsplit("/", n=1).str[-1].isin(have_uids)]
    return rest.head(need) if sample_seed is None else rest.sample(n=min(need, len(rest)), random_state=sample_seed)


def get_objects(dataset: str, count: int, download_dir: Path, refresh: bool, sample_seed: int | None = None, have_uids: set[str] = frozenset()) -> pd.DataFrame:
    """Return `count` sketchfab objects (see pick()) as a fileIdentifier/sha256/source frame."""
    if dataset == "trellis-500k":
        trellis = get_trellis_500k_csv(download_dir, refresh)
        subset = pick(trellis, count, sample_seed, have_uids, "file_identifier")
        return pd.DataFrame(
            {
                "fileIdentifier": subset["file_identifier"],
                "sha256": subset["sha256"],
                "source": "sketchfab",
            }
        )

    print(f"Loading full Objaverse-XL sketchfab annotations into {download_dir}...")
    annotations = oxl.get_annotations(download_dir=str(download_dir), refresh=refresh)
    subset = pick(annotations[annotations["source"] == "sketchfab"], count, sample_seed, have_uids, "fileIdentifier")
    return subset[["fileIdentifier", "sha256", "source"]].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser()

    have_uids = set()
    if args.have_file is not None:
        have_uids = {Path(line.strip()).stem for line in open(args.have_file) if line.strip()}

    objects = get_objects(args.dataset, args.count, download_dir, args.refresh, args.sample_seed, have_uids)
    print(f"Downloading {len(objects)} Sketchfab meshes ({args.dataset}; {len(have_uids)} already downloaded) into {download_dir}...")

    paths = oxl.download_objects(objects=objects, download_dir=str(download_dir), processes=args.processes)

    print(f"Done. {len(paths)} meshes cached under {download_dir}")


if __name__ == "__main__":
    main()
