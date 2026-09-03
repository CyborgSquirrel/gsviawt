#!/usr/bin/env python3
"""Modal deployment of the World Tracing layer-inference pipeline.

Renders a mesh with capture_turntable.py (headless Blender via rpyc, see
render_server.py) and then runs wt_infer_layers.py's diffusion model on one
view of the result. Reuses this repo's own Dockerfile unchanged, so the
container matches the docker-compose dev image (same Blender install, same
uv-managed venv, same `wt` package).

Caching mirrors docker-compose.yml's volume layout, just backed by Modal
Volumes instead of bind mounts:
  - mounts/huggingface:/home/user/.cache/huggingface -> "wt-hf-cache" volume
    (checkpoint.py's build_model_and_load_ckpt uses hf_hub_download with the
    default cache dir, so this is a drop-in cache for model weights)
  - the objaverse glb the mesh_strategy config points at -> "wt-objaverse-data"
    volume mounted at /app/data, matching conf/mesh_strategy/list.yaml's
    hardcoded /app/data/objaverse/hf-objaverse-v1/glbs/... path. Downloaded
    lazily from the same HF dataset repo the `objaverse` package itself
    pulls from (allenai/objaverse) if not already cached on the volume.
    Note: the "objaverse/" path segment is local-only (mirrors the
    `objaverse` package's own ~/.objaverse/hf-objaverse-v1/... layout) --
    the HF repo's internal paths start at "hf-objaverse-v1/...", so it is
    NOT part of the filename passed to hf_hub_download below.
  - render + inference output (renders.h5, *.wt.h5) -> "wt-outputs" volume
    mounted at /app/bla, matching conf/config.yaml's output_path default.

Usage:
    modal run modal_app.py
    modal run modal_app.py --view-index 2
    modal run modal_app.py --glb-rel-path hf-objaverse-v1/glbs/000-147/other.glb
"""

import os
import pathlib

import modal

# Relative to both /app/data/objaverse (local mesh_strategy path) and the
# allenai/objaverse HF dataset repo root (same "hf-objaverse-v1/..." layout
# in both places -- see the module docstring).
DEFAULT_GLB_REL_PATH = "hf-objaverse-v1/glbs/000-147/405f47d6ce6d481d94f54800ee913fa4.glb"
HF_DATASET_REPO = "allenai/objaverse"

app = modal.App("gsviawt-wt-infer")

image = modal.Image.from_dockerfile(
  "Dockerfile",
  build_args={"XUID": "1000", "XGID": "1000"},
)

hf_cache_volume = modal.Volume.from_name("wt-hf-cache", create_if_missing=True)
objaverse_volume = modal.Volume.from_name("wt-objaverse-data", create_if_missing=True)
outputs_volume = modal.Volume.from_name("wt-outputs", create_if_missing=True)


@app.function(
  image=image,
  gpu="L4",
  timeout=30 * 60,
  volumes={
    "/home/user/.cache/huggingface": hf_cache_volume,
    "/app/data": objaverse_volume,
    "/app/bla": outputs_volume,
  },
)
def infer_layers(glb_rel_path: str = DEFAULT_GLB_REL_PATH, view_index: int = 0, ckpt: str = "r75b") -> bytes:
  import subprocess

  from huggingface_hub import hf_hub_download

  glb_abs_path = f"/app/data/objaverse/{glb_rel_path}"
  if not os.path.exists(glb_abs_path):
    print(f"[modal] {glb_abs_path} not cached, downloading from hf://{HF_DATASET_REPO}/{glb_rel_path}")
    hf_hub_download(
      repo_id=HF_DATASET_REPO,
      repo_type="dataset",
      filename=glb_rel_path,
      local_dir="/app/data/objaverse",
    )
    objaverse_volume.commit()

  subprocess.run(["python", "capture_turntable.py"], cwd="/app", check=True)
  outputs_volume.commit()

  subprocess.run(
    ["python", "wt_infer_layers.py", "/app/bla/renders.h5", str(view_index), "--ckpt", ckpt, "--config", ckpt],
    cwd="/app", check=True,
  )
  hf_cache_volume.commit()
  outputs_volume.commit()

  out_path = f"/app/bla/renders.h5.view{view_index}.wt.h5"
  return pathlib.Path(out_path).read_bytes()


@app.local_entrypoint()
def main(glb_rel_path: str = DEFAULT_GLB_REL_PATH, view_index: int = 0, ckpt: str = "r75b", out: str = "wt_output.h5"):
  data = infer_layers.remote(glb_rel_path, view_index, ckpt)
  pathlib.Path(out).write_bytes(data)
  print(f"[modal] wrote {out} ({len(data)} bytes)")
