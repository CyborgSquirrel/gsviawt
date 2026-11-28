"""Fetch Objaverse (Sketchfab) GLBs by uid into one directory, kill-safe, for build_dataset.py's downloader.

    index = local_index(["/mnt/hdd/objaverse/hf-objaverse-v1/glbs"])   # uid -> GLB already on disk
    hf = hf_paths("~/.objaverse")                                      # uid -> path in the HF objaverse repo
    fetch(uid, sha256, "<root>/glbs", index, hf)                       # -> "<root>/glbs/<uid>.glb"

fetch takes a GLB from a local pool (hard link on the same filesystem, else a copy) when one is there with the expected
sha256, else downloads it from the Hugging Face objaverse repo (the objaverse package's uid -> path index,
objaverse.xl.SketchfabDownloader). Written to <dest>/<uid>.glb.<pid>.tmp, sha256-checked against the expected value (the
TRELLIS CSV's), fsynced, renamed; a file that is there is never fetched again. It raises PermanentError when the object can't
be had (not in the index, HTTP 4xx, sha256 mismatch) and anything else (network down, HTTP 5xx, timeouts) as it is: the
caller retries those later and never gives up on a mesh for a transient error (an outage must not reject meshes).
"""

import glob
import hashlib
import os
import shutil
import urllib.error
import urllib.request

from util import fsync_path

HF_BASE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"


class PermanentError(Exception):
  """The object can't be fetched (gone / changed): retrying won't help."""


def glb_path(dest, uid):
  return os.path.join(dest, f"{uid}.glb")


def local_index(dirs):
  """{uid: path} of the GLBs under dirs (hf-objaverse-v1 layout <dir>/<group>/<uid>.glb, or flat <dir>/<uid>.glb); the
  first dir holding a uid wins."""
  index = {}
  for d in dirs:
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(str(d)), "*.glb")) +
                    glob.glob(os.path.join(os.path.expanduser(str(d)), "*", "*.glb"))):
      index.setdefault(os.path.basename(p)[:-len(".glb")], p)
  return index


def hf_paths(objaverse_dir):
  """{uid: path in the HF objaverse repo} (objaverse's object-paths.json.gz, cached in objaverse_dir)."""
  import objaverse.xl as oxl
  return oxl.SketchfabDownloader._get_object_paths(download_dir=os.path.expanduser(str(objaverse_dir)))


def sha256_of(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      h.update(block)
  return h.hexdigest()


def _download(url, tmp, timeout):
  try:
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
      shutil.copyfileobj(r, f, 1 << 20)
  except urllib.error.HTTPError as e:
    if 400 <= e.code < 500:
      raise PermanentError(f"HTTP {e.code} for {url}") from e
    raise


def fetch(uid, sha256, dest, index, hf, timeout=120):
  """<dest>/<uid>.glb with the given sha256 (see the module docstring) -> its path."""
  final = glb_path(dest, uid)
  if os.path.exists(final):
    return final
  os.makedirs(dest, exist_ok=True)
  tmp = f"{final}.{os.getpid()}.tmp"
  try:
    local = index.get(uid)
    if local is not None:
      try:
        os.link(local, tmp)
      except OSError:
        shutil.copyfile(local, tmp)
      if sha256_of(tmp) != sha256:   # a local copy that differs: take the repo's instead
        os.remove(tmp)
        local = None
    if local is None:
      if uid not in hf:
        raise PermanentError(f"{uid} is not in objaverse's object-paths index")
      _download(f"{HF_BASE}/{hf[uid]}", tmp, timeout)
      got = sha256_of(tmp)
      if got != sha256:
        raise PermanentError(f"sha256 mismatch: got {got}, the CSV says {sha256}")
    fsync_path(tmp)
    os.replace(tmp, final)
    fsync_path(dest, directory=True)
  finally:
    if os.path.exists(tmp):
      os.remove(tmp)
  return final
