#!/usr/bin/env python3
"""CLIP ViT-L/14 image embeddings + the LAION improved-aesthetic-predictor
MLP head (https://github.com/christophschuhmann/improved-aesthetic-predictor),
computed over the "images" dataset of an existing render_objaverse.py h5 and
written back into it as "clip_embedding" (N x 768 float32) and, optionally,
"aesthetic_score" (N float32).

Runs in the project's normal venv -- *not* inside Blender's bundled Python,
which only has h5py/hydra-core installed (see render_objaverse.py's
docstring). render_objaverse.py's own `compute_clip`/`compute_aesthetic`
toggles no longer call this script: they spawn clip_worker.py/
aesthetic_worker.py as long-lived workers instead and score views live,
during the render (see ipc.py, clip_worker.py). This script remains a
standalone batch tool for scoring an h5 that was rendered without
compute_clip, or re-scoring one with a different background; `load_clip_model`/
`embed_batch`/`aesthetic_scores` below are the same functions the live
workers use, so both paths score identically. Run by hand against an
existing h5:

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python \\
        clip_score.py output_path=/app/bla/objaverse_renders.h5

Uses OpenAI's original ViT-L/14 weights (not an OpenCLIP LAION-2B variant --
the aesthetic MLP was trained specifically on those embeddings), with the
QuickGELU activation OpenAI's checkpoint expects (open_clip's plain
"ViT-L-14" tag defaults to GELU and silently mismatches otherwise).

Renders are RGBA with a transparent background; CLIP has never seen
transparency, so each frame is alpha-composited onto `cfg.background` first.
That choice measurably changes both the embedding and the aesthetic score
(cosine similarity as low as ~0.87 between white- and black-composited
renders of the same view) -- it is not a cosmetic default.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from util import timed  # noqa: E402

log = logging.getLogger("clip_score")

_AESTHETIC_WEIGHTS_URL = (
  "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
  "raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
_AESTHETIC_CACHE = os.path.expanduser(
  "~/.cache/laion-aesthetic/sac+logos+ava1-l14-linearMSE.pth")


class AestheticMLP(torch.nn.Module):
  """Matches the checkpoint's architecture exactly (input: a 768-d,
  L2-normalized CLIP ViT-L/14 embedding)."""

  def __init__(self, input_size=768):
    super().__init__()
    self.layers = torch.nn.Sequential(
      torch.nn.Linear(input_size, 1024),
      torch.nn.Dropout(0.2),
      torch.nn.Linear(1024, 128),
      torch.nn.Dropout(0.2),
      torch.nn.Linear(128, 64),
      torch.nn.Dropout(0.1),
      torch.nn.Linear(64, 16),
      torch.nn.Linear(16, 1),
    )

  def forward(self, x):
    return self.layers(x)


def load_aesthetic_mlp(device):
  if not os.path.exists(_AESTHETIC_CACHE):
    os.makedirs(os.path.dirname(_AESTHETIC_CACHE), exist_ok=True)
    log.info("downloading LAION aesthetic predictor weights to %s", _AESTHETIC_CACHE)
    import urllib.request
    urllib.request.urlretrieve(_AESTHETIC_WEIGHTS_URL, _AESTHETIC_CACHE)

  state_dict = torch.load(_AESTHETIC_CACHE, map_location="cpu", weights_only=True)
  got_shape = tuple(state_dict["layers.0.weight"].shape)
  if got_shape != (1024, 768):
    raise ValueError(
      f"aesthetic MLP checkpoint at {_AESTHETIC_CACHE} has layers.0.weight "
      f"shape {got_shape}, expected (1024, 768) -- looks like a bad/partial "
      "download; delete the file and retry")

  mlp = AestheticMLP(768)
  mlp.load_state_dict(state_dict)
  return mlp.to(device).eval()


def resolve_device(device_cfg):
  if device_cfg == "auto":
    return "cuda" if torch.cuda.is_available() else "cpu"
  return device_cfg


def load_clip_model(device):
  """OpenAI's ViT-L/14 CLIP weights via open_clip's QuickGELU tag (see
  module docstring for why not the plain "ViT-L-14" tag). Shared by the
  standalone batch path below and clip_worker.py. Returns (model, preprocess)."""
  import open_clip

  with timed("load CLIP ViT-L/14 (openai)"):
    model, _, preprocess = open_clip.create_model_and_transforms(
      "ViT-L-14-quickgelu", pretrained="openai", device=device)
    model.eval()
  return model, preprocess


@torch.no_grad()
def embed_batch(model, preprocess, rgb_uint8, device):
  """(N, H, W, 3) uint8 RGB -> (N, 768) float32 CLIP embeddings,
  un-normalized (L2-normalize before feeding the aesthetic MLP -- see
  aesthetic_scores)."""
  from PIL import Image

  batch = torch.stack(
    [preprocess(Image.fromarray(im, mode="RGB")) for im in rgb_uint8]
  ).to(device)
  return model.encode_image(batch).float().cpu().numpy()


@torch.no_grad()
def aesthetic_scores(mlp, emb):
  """(N, 768) float32 CLIP embeddings -> (N,) float32 LAION-Aesthetics V2
  scores. L2-normalizes internally (the MLP was trained on normalized
  embeddings)."""
  device = next(mlp.parameters()).device
  t = torch.from_numpy(emb).to(device)
  normed = t / t.norm(dim=-1, keepdim=True)
  return mlp(normed).squeeze(-1).cpu().numpy()


def composite(images, background):
  """(N, H, W, 4) uint8 RGBA -> (N, H, W, 3) uint8 RGB, alpha-composited onto
  `background` (an RGB triple in [0, 1]) -- Blender's PNG output uses
  straight (non-premultiplied) alpha, so this is the standard `over` blend.
  Older render_objaverse.py h5s only have a 3-channel "images" (already
  opaque RGB, no film_transparent pass); those are returned unchanged."""
  if images.shape[-1] == 3:
    return images
  if images.shape[-1] != 4:
    raise ValueError(f"'images' has {images.shape[-1]} channels, expected 3 (RGB) or 4 (RGBA)")
  rgb = images[..., :3].astype(np.float32)
  alpha = images[..., 3:4].astype(np.float32) / 255.0
  bg = np.asarray(list(background), np.float32) * 255.0
  out = rgb * alpha + bg * (1.0 - alpha)
  return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def ensure_dataset(hf, name, n, item_shape, dtype):
  """(Re)create a top-level (n, *item_shape) dataset in `hf`. Re-running this
  script on an h5 that already has the dataset overwrites it -- no silent
  append/skip."""
  if name in hf:
    log.warning("%s already present in %s -- overwriting", name, hf.filename)
    del hf[name]
  return hf.create_dataset(name, shape=(n, *item_shape), dtype=dtype)


@hydra.main(version_base=None, config_path="conf", config_name="clip_score")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO)

  device = resolve_device(str(cfg.device))
  log.info("device: %s", device)

  model, preprocess = load_clip_model(device)
  aesthetic_mlp = load_aesthetic_mlp(device) if cfg.compute_aesthetic else None

  with h5py.File(cfg.output_path, "a") as hf:
    if "images" not in hf:
      raise ValueError(
        f"{cfg.output_path} has no 'images' dataset -- render with render=true "
        "first (clip scoring needs the RGB pass)")
    images = hf["images"]
    n = images.shape[0]

    ds_emb = ensure_dataset(hf, "clip_embedding", n, (768,), np.float32)
    ds_aes = (ensure_dataset(hf, "aesthetic_score", n, (), np.float32)
              if aesthetic_mlp is not None else None)

    batch_size = int(cfg.batch_size)
    with timed(f"score {n} views"):
      for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rgb = composite(images[start:end], cfg.background)
        emb = embed_batch(model, preprocess, rgb, device)
        ds_emb[start:end] = emb

        if aesthetic_mlp is not None:
          ds_aes[start:end] = aesthetic_scores(aesthetic_mlp, emb)

        log.info("scored %d/%d", end, n)

  log.info("wrote clip_embedding%s to %s",
            " + aesthetic_score" if aesthetic_mlp is not None else "",
            cfg.output_path)


if __name__ == "__main__":
  main()
