"""KID image features (eval_diffusion.py's kid option; kid_compare.py scores them): DINOv2 ViT-L/14 CLS tokens (1024-d) of renders
composited onto white, with dgm-eval's (layer6ai-labs) DINOv2 encoder preprocessing in its --clean_resize mode:
  white      each image over a white background with its OWN alpha: the dataset images are straight (unpremultiplied) RGBA ->
             rgb * a + (1 - a); the gsplat renders are premultiplied over black -> rgb + (1 - a)
  uint8      quantized (round) to 8 bits, as an image file would be; no crop
  resize     dgm_eval.resizer.pil_resize: each channel a float32 PIL image ('F'), bicubic to 224 x 224, clipped to [0, 255], / 255
  normalize  ImageNet mean / std (torchvision Normalize)
  features   the hub model's forward(): the final-norm CLS token (not L2-normalized), fp32 with autocast off
The model is torch.hub's facebookresearch/dinov2 'dinov2_vitl14' loaded from a LOCAL copy of the hub repo (pinned commit, no network)
and the weights in torch.hub's checkpoints dir (dinov2_vitl14_pretrain.pth). FEATURE_TAG names this preprocessing + model: the GT
feature cache is keyed by it."""
import os

import numpy as np
import torch
from PIL import Image

DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
FEATURE_TAG = f"dinov2_vitl14_{DINOV2_COMMIT[:7]}_white_u8_pil224"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def default_repo():
  """<torch hub dir>/facebookresearch_dinov2_<commit[:7]>: the hub repo at DINOV2_COMMIT (codeload.github.com tarball, unpacked)."""
  return os.path.join(torch.hub.get_dir(), f"facebookresearch_dinov2_{DINOV2_COMMIT[:7]}")


def load_dinov2(repo=None, device="cuda"):
  repo = repo or default_repo()
  if not os.path.isfile(os.path.join(repo, "hubconf.py")):
    raise SystemExit(f"kid: no DINOv2 hub repo at {repo!r} (download github.com/facebookresearch/dinov2 at {DINOV2_COMMIT} there)")
  return torch.hub.load(repo, "dinov2_vitl14", source="local").to(device).eval()


def white_uint8(rgb, alpha, premultiplied):
  """(V, 3, H, W) float rgb in [0, 1] and (V, 1, H, W) alpha -> (V, H, W, 3) uint8 numpy, over white with that alpha. premultiplied:
  the gsplat renders (rgb already carries alpha); else straight RGBA (the dataset images)."""
  rgb, alpha = rgb.float().clamp(0, 1), alpha.float().clamp(0, 1)
  img = (rgb if premultiplied else rgb * alpha) + (1 - alpha)
  return (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def pil_resize(u8, size=224):
  """One (H, W, 3) uint8 image -> (3, size, size) float32 in [0, 1]: dgm_eval.resizer.pil_resize (per channel 'F' bicubic, clip)."""
  x = u8.astype(np.float32)
  # a float32 2-D array is a mode 'F' image (fromarray's mode= argument is deprecated)
  ch = [np.asarray(Image.fromarray(np.ascontiguousarray(x[:, :, c])).resize((size, size), resample=Image.BICUBIC)).clip(0, 255)
        for c in range(3)]
  return np.stack(ch).astype(np.float32) / 255


def preprocess(u8):
  """(V, H, W, 3) uint8 -> (V, 3, 224, 224) float32 tensor, resized and ImageNet-normalized."""
  x = np.stack([pil_resize(im) for im in u8])
  return torch.from_numpy((x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]).float()


def features(model, u8, device="cuda"):
  """(V, H, W, 3) uint8 -> (V, 1024) float32 numpy CLS features, fp32 with autocast off."""
  with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
    return model(preprocess(u8).to(device)).float().cpu().numpy()


class GTCache:
  """GT features per (render file, view row): <dir>/<FEATURE_TAG>/<file name>_<file size>/<row>.npy (written atomically, so
  concurrent evals can share the dir). dir None: no cache."""

  def __init__(self, cache_dir):
    self.dir = os.path.join(cache_dir, FEATURE_TAG) if cache_dir else None

  def _path(self, source_h5, view):
    return os.path.join(self.dir, f"{os.path.basename(source_h5)}_{os.path.getsize(source_h5)}", f"{int(view)}.npy")

  def get(self, source_h5, views):
    """(V, 1024) or None if any view is missing."""
    if self.dir is None:
      return None
    paths = [self._path(source_h5, v) for v in views]
    if not all(os.path.exists(p) for p in paths):
      return None
    return np.stack([np.load(p) for p in paths])

  def put(self, source_h5, views, feats):
    if self.dir is None:
      return
    for v, f in zip(views, feats):
      p = self._path(source_h5, v)
      os.makedirs(os.path.dirname(p), exist_ok=True)
      tmp = f"{p}.{os.getpid()}.tmp.npy"
      np.save(tmp, f)
      os.replace(tmp, p)
