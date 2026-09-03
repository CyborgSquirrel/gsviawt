#!/usr/bin/env python3

"""Drive the Blender rpyc render server (render.py) to capture turntables of
a configurable set of meshes. Configured via Hydra -- see conf/config.yaml.
"""

import contextlib as ctl
import json
import os
import signal
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from rpyc.utils.factory import unix_connect

from util import LazyDataset, timed

BLENDER_BIN = "/opt/blender/blender"
RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_server.py")
SOCKET_PATH = "/tmp/blender.sock"

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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
  global _proc

  signal.signal(signal.SIGINT, _signal_handler)
  signal.signal(signal.SIGTERM, _signal_handler)

  if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

  os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

  args = [BLENDER_BIN, "--background", "--python", RENDER_SCRIPT, "--", SOCKET_PATH]

  _proc = subprocess.Popen(args)

  try:
    wait_for_socket(SOCKET_PATH)

    conn = unix_connect(SOCKET_PATH, config={"sync_request_timeout": 600})
    try:
      conn.root.init(device=cfg.device)

      view_strategy = hydra.utils.instantiate(cfg.view_strategy)
      meshes = hydra.utils.instantiate(cfg.mesh_strategy).meshes()

      image_kwargs = {
        "chunks": (1, cfg.height, cfg.width, 3),  # one chunk per image -> compressed independently
        "compression": "gzip",
        "compression_opts": 4,
      }
      depth_kwargs = {
        "chunks": (1, cfg.height, cfg.width, cfg.max_peel_layers),  # one chunk per view -> compressed independently
        "compression": "gzip",
        "compression_opts": 4,
      }

      with ctl.ExitStack() as stack:
        tmp_dir = stack.enter_context(TemporaryDirectory())

        hf = stack.enter_context(h5py.File(cfg.output_path, "w"))

        hf.attrs["config_json"] = json.dumps(OmegaConf.to_container(cfg, resolve=True))
        hf.create_dataset("mesh_paths", data=meshes, dtype=h5py.string_dtype(encoding="utf-8"))

        images_ds = None
        if cfg.render:
          images_ds = stack.enter_context(LazyDataset(hf, "images", dataset_kwargs=image_kwargs))
        pose_ds     = stack.enter_context(LazyDataset(hf, "camera_pose"))
        intr_ds     = stack.enter_context(LazyDataset(hf, "camera_intrinsics"))
        depth_ds    = stack.enter_context(LazyDataset(hf, "depth_peel", dataset_kwargs=depth_kwargs))
        mesh_idx_ds = stack.enter_context(LazyDataset(hf, "mesh_index"))

        for mesh_idx, mesh_path in enumerate(meshes):
          conn.root.reset(mesh_path)

          for tilt_deg, azimuth_deg, view_dir in view_strategy.views():
            tmp_path = os.path.join(tmp_dir, "output.png")
            tmp_depth_path = os.path.join(tmp_dir, "depth.npy")
            view_label = f"mesh={mesh_idx} tilt={tilt_deg:.1f} az={azimuth_deg:.1f}"

            with timed(f"{view_label} rpyc_render"):
              result = conn.root.render(
                width=cfg.width, height=cfg.height, path=tmp_path, view_dir=view_dir,
                max_layers=cfg.max_peel_layers, depth_path=tmp_depth_path,
                capture_rgb=cfg.render,
              )

            with timed(f"{view_label} depth_load"):
              depth_volume = np.load(tmp_depth_path)
              os.remove(tmp_depth_path)

            if cfg.render:
              with timed(f"{view_label} image_decode"):
                image = np.asarray(Image.open(tmp_path).convert("RGB"), dtype=np.uint8)
                os.remove(tmp_path)

              with timed(f"{view_label} image_write"):
                images_ds.append(image)

            with timed(f"{view_label} depth_write"):
              depth_ds.append(depth_volume)

            with timed(f"{view_label} other_write"):
              pose_ds.append(np.array(result["pose_matrix"], dtype=np.float32))
              intr_ds.append(np.array(result["intrinsics_matrix"], dtype=np.float32))
              mesh_idx_ds.append(np.int64(mesh_idx))
    finally:
      conn.close()
  finally:
    cleanup()


if __name__ == "__main__":
  main()
