#!/usr/bin/env python3
"""Print local glb paths for the first N Sketchfab meshes of a dataset
(trellis-500k or the full objaverse-xl sketchfab annotations), resolved
against files already fetched by download_objaverse_sketchfab.py.

Reuses that script's dataset selection (get_objects) and objaverse-xl's own
uid -> hf-objaverse-v1 path mapping, so this never needs to know the
downloader's internal layout. Meshes not yet downloaded are skipped with a
warning on stderr.

Usage:
    python scripts/sketchfab_sample_paths.py --dataset trellis-500k --count 25
"""

import argparse
import contextlib
import sys
from pathlib import Path

from download_objaverse_sketchfab import get_objects
from objaverse.xl.sketchfab import SketchfabDownloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["trellis-500k", "objaverse-xl"], default="trellis-500k")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--download-dir", default="~/.objaverse")
    return parser.parse_args()


def resolve_local_paths(dataset: str, count: int, download_dir: Path) -> list[str]:
    # get_objects() prints progress (e.g. "Using cached ...") straight to
    # stdout; keep that off the path list stdout callers capture.
    with contextlib.redirect_stdout(sys.stderr):
        objects = get_objects(dataset, count, download_dir, refresh=False)
    uid_to_hf_path = SketchfabDownloader._get_object_paths(download_dir=str(download_dir))

    paths = []
    for file_identifier in objects["fileIdentifier"]:
        uid = file_identifier.split("/")[-1]
        hf_path = uid_to_hf_path.get(uid)
        local_path = download_dir / "hf-objaverse-v1" / hf_path if hf_path else None
        if local_path is None or not local_path.exists():
            print(f"skipping {uid} (not downloaded): {file_identifier}", file=sys.stderr)
            continue
        paths.append(str(local_path))
    return paths


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser()
    paths = resolve_local_paths(args.dataset, args.count, download_dir)
    print(f"resolved {len(paths)}/{args.count} meshes for {args.dataset}", file=sys.stderr)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
