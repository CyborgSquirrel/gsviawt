#!/usr/bin/env python3

"""Drive the Blender rpyc render server (render.py) to capture a turntable
of a hardcoded mesh: 8 azimuths equidistant around the z axis, x3 tilts
(high, mid, low) = 24 renders total.
"""

import contextlib as ctl
import math
import os
import signal
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from PIL import Image
from rpyc.utils.factory import unix_connect

from util import LazyDataset

BLENDER_BIN = "/opt/blender/blender"
RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render.py")
SOCKET_PATH = "/tmp/blender.sock"

MESH_PATH = "/app/data/objaverse/hf-objaverse-v1/glbs/000-147/405f47d6ce6d481d94f54800ee913fa4.glb"
OUTPUT_DIR = "/app/bla"
HDF5_PATH = os.path.join(OUTPUT_DIR, "renders.h5")

# WIDTH, HEIGHT = 504, 504
WIDTH, HEIGHT = 112, 112
NUM_AZIMUTHS = 2
TILTS_DEG = [45, 15, -15]  # high, mid, low
IMAGE_DATASET_KWARGS = {
  "chunks": (1, HEIGHT, WIDTH, 3),  # one chunk per image -> compressed independently
  "compression": "gzip",
  "compression_opts": 4,
}

_proc = None


def cleanup():
  global _proc
  if _proc is not None and _proc.poll() is None:
    _proc.terminate()
    try:
      _proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
      _proc.kill()
      _proc.wait()


def _signal_handler(signum, frame):
  cleanup()
  sys.exit(1)


def wait_for_socket(path, timeout=60):
  start = time.time()
  while not os.path.exists(path):
    if time.time() - start > timeout:
      raise TimeoutError(f"Timed out waiting for socket at {path}")
    time.sleep(0.2)


def view_dir_for(azimuth_rad, tilt_rad):
  return (
    math.cos(tilt_rad) * math.cos(azimuth_rad),
    math.cos(tilt_rad) * math.sin(azimuth_rad),
    math.sin(tilt_rad),
  )


def main():
  global _proc

  signal.signal(signal.SIGINT, _signal_handler)
  signal.signal(signal.SIGTERM, _signal_handler)

  if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

  os.makedirs(OUTPUT_DIR, exist_ok=True)

  bg = True

  # Build proc args
  args = []
  args += [BLENDER_BIN]
  if bg:
    args += ["--background"]
  args += ["--python", RENDER_SCRIPT, "--", SOCKET_PATH]

  _proc = subprocess.Popen(args)

  try:
    wait_for_socket(SOCKET_PATH)

    conn = unix_connect(SOCKET_PATH, config={"sync_request_timeout": 600})
    try:
      conn.root.reset(MESH_PATH)

      with ctl.ExitStack() as stack:
        tmp_dir = stack.enter_context(TemporaryDirectory())

        hf        = stack.enter_context(h5py.File(HDF5_PATH, "w"))
        images_ds = stack.enter_context(LazyDataset(hf, "images", dataset_kwargs=IMAGE_DATASET_KWARGS))
        pose_ds   = stack.enter_context(LazyDataset(hf, "camera_pose"))

        for tilt_deg in TILTS_DEG:
          tilt_rad = math.radians(tilt_deg)
          for az_idx in range(NUM_AZIMUTHS):
            azimuth_rad = az_idx * 2 * math.pi / NUM_AZIMUTHS
            # azimuth_deg = math.degrees(azimuth_rad)
            view_dir = view_dir_for(azimuth_rad, tilt_rad)
            # print(f"Rendering azimuth={azimuth_deg:.0f} deg, tilt={tilt_deg} deg")

            tmp_path = os.path.join(tmp_dir, "output.png")
            result = conn.root.render(width=WIDTH, height=HEIGHT, path=tmp_path, view_dir=view_dir)
            image = np.asarray(Image.open(tmp_path).convert("RGB"), dtype=np.uint8)
            os.remove(tmp_path)

            images_ds.append(image)
            pose_ds.append(np.array(result["pose_matrix"], dtype=np.float32))
    finally:
      conn.close()
  finally:
    cleanup()


if __name__ == "__main__":
  main()
