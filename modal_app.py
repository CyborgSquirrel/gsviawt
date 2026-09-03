#!/usr/bin/env python3
"""Modal deployment of the World Tracing layer-inference pipeline.

Renders a mesh with capture_turntable.py (headless Blender via rpyc, see
render_server.py) and then runs wt_infer_layers.py's diffusion model on one
view of the result. Reuses this repo's own Dockerfile unchanged, so the
container matches the docker-compose dev image (same Blender install, same
uv-managed venv, same `wt` package).

Caching mirrors docker-compose.yml's volume layout, just backed by Modal
Volumes instead of bind mounts:
  - mounts/huggingface:/home/user/.cache/huggingface -> "hf-cache" volume
    (checkpoint.py's build_model_and_load_ckpt uses hf_hub_download with the
    default cache dir, so this is a drop-in cache for model weights)
  - mounts/objaverse:/home/user/.objaverse -> "objaverse-cache" volume.
    The target mesh is fetched lazily via objaverse.xl (same API
    scripts/download_objaverse_sketchfab.py uses), which downloads straight
    into ~/.objaverse/hf-objaverse-v1/glbs/<uid_prefix>/<uid>.glb.
  - render + inference output (renders.h5, *.wt.h5) -> "outputs" volume
    mounted at /app/bla, matching conf/config.yaml's output_path default.

conf/mesh_strategy/list.yaml hardcodes a /app/data/... mesh path that this
deployment doesn't populate (that directory backed a plain host bind mount
in docker-compose, not something any script here downloads); the actual
downloaded path under ~/.objaverse is passed to capture_turntable.py via a
Hydra CLI override instead.

Usage:
    modal run modal_app.py
    modal run modal_app.py --view-index 2
    modal run modal_app.py --uid 405f47d6ce6d481d94f54800ee913fa4
"""

import pathlib

import modal

DEFAULT_UID = "405f47d6ce6d481d94f54800ee913fa4"

app = modal.App("gsviawt-wt-infer")

image = modal.Image.from_dockerfile(
  "Dockerfile",
  build_args={"XUID": "1000", "XGID": "1000"},
)

hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
objaverse_volume = modal.Volume.from_name("objaverse-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("outputs", create_if_missing=True)


def _download_mesh(uid: str) -> str:
  """Fetch one Sketchfab mesh by uid via objaverse.xl into ~/.objaverse,
  mirroring scripts/download_objaverse_sketchfab.py's download call but
  filtered down to a single object instead of the first N."""
  import objaverse.xl as oxl

  file_identifier = f"https://sketchfab.com/3d-models/{uid}"
  annotations = oxl.get_annotations()
  match = annotations[annotations["fileIdentifier"] == file_identifier]
  if match.empty:
    raise ValueError(f"uid {uid!r} not found in objaverse-xl annotations")

  paths = oxl.download_objects(objects=match, processes=1)
  return paths[file_identifier]


@app.function(
  image=image,
  gpu="L4",
  timeout=30 * 60,
  volumes={
    "/home/user/.cache/huggingface": hf_cache_volume,
    "/home/user/.objaverse": objaverse_volume,
    "/app/bla": outputs_volume,
  },
)
def infer_layers(uid: str = DEFAULT_UID, view_index: int = 0, ckpt: str = "r75b") -> bytes:
  import subprocess

  mesh_path = _download_mesh(uid)
  objaverse_volume.commit()

  subprocess.run(
    ["python", "capture_turntable.py", f"mesh_strategy.paths=[{mesh_path}]"],
    cwd="/app", check=True,
  )
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
def main(uid: str = DEFAULT_UID, view_index: int = 0, ckpt: str = "r75b", out: str = "wt_output.h5"):
  data = infer_layers.remote(uid, view_index, ckpt)
  pathlib.Path(out).write_bytes(data)
  print(f"[modal] wrote {out} ({len(data)} bytes)")
