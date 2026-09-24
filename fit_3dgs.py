#!/usr/bin/env python3
"""Optimize a 3D Gaussian Splatting model from one render HDF5, porting the
original Inria training loop (github.com/graphdeco-inria/gaussian-splatting,
cloned locally at ~/projects/gaussian-splatting) on top of this repo's own
data format and rendering (gsplat, not diff-gaussian-rasterization).

Ported from the reference: SH color (features_dc + up to `max_sh_degree` SH
bands, active_sh_degree incremented every `sh_degree_interval` iters), the
per-parameter Adam learning rates and L1 + D-SSIM loss (`lambda_dssim`),
periodic opacity reset, periodic pruning of low-opacity / oversized Gaussians,
and one-random-camera-per-step sampling (reshuffled every epoch).

Deliberately NOT ported: densification (`densify_and_clone` /
`densify_and_split`) and xyz optimization. The reference grows its point cloud
by cloning/splitting Gaussians to new positions and freely moves every
Gaussian's center; this repo instead seeds Gaussians 1:1 from the primary
view's depth-peel pixels (see fit_gsplat.py, whose config and seeding scheme
this one shares) and keeps that pixel alignment for the whole run, so centers
are frozen and never added to -- only ever pruned. Pruning is therefore
remove-only, reusing gsplat's `strategy.ops.remove`/`reset_opa` (the same
primitives gsplat's own `DefaultStrategy` -- itself a from-scratch port of the
reference's densification -- calls internally), just without the grow half.

Output is written next to the input as:
  <h5>.fit3dgs.view<primary>.h5   -- Gaussian attributes as (H, W, 6, .) grids
      (NaN marks an empty slot -- either never seeded, or pruned during
      training -- same layout as fit_gsplat.py's output / wt_infer_layers'
      `points`), a `layer_valid` mask, and the views used (cameras + GT).
  <h5>.fit3dgs.view<primary>.ply  -- standard 3DGS point cloud with full SH (if write_ply)
  <h5>.fit3dgs.view<primary>.val/ -- gt|render comparison PNGs (if val_every)

Config is Hydra (`conf/fit_3dgs.yaml`); run in the container venv, e.g.

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python fit_3dgs.py \\
        hdf5_path=/app/bla/obj_lite.h5 primary=0 'secondary=[1,2,3]' iters=5000

gsplat JIT-compiles CUDA kernels on first import; this module points it at the
pip `nvidia-cuda-nvcc` toolchain (the base image has no system `nvcc`).
"""

import json
import logging
import os
import shutil
import sys
import sysconfig
import tempfile


def _setup_cuda_toolchain() -> None:
  """Make the pip-installed CUDA toolkit (`nvidia-cuda-nvcc` etc.) usable by
  gsplat's `torch.utils.cpp_extension` JIT build. Copied from fit_gsplat.py,
  which needs the same env setup -- see its copy for the full explanation."""
  if os.environ.get("CUDA_HOME") and os.path.exists(
      os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")):
    return
  for libdir in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
    cuda_home = os.path.join(libdir, "nvidia", "cu13")
    if not os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
      continue
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    lib = os.path.join(cuda_home, "lib")
    link, real = os.path.join(lib, "libcudart.so"), os.path.join(lib, "libcudart.so.13")
    if os.path.exists(real) and not os.path.exists(link):
      try:
        os.symlink("libcudart.so.13", link)
      except OSError:
        pass
    return


_setup_cuda_toolchain()

import h5py  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
from einops import rearrange, reduce, repeat  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_pointcloud import unproject_depth_peel  # noqa: E402
from fit_gsplat import SH_C0, make_viewmats, ssim, load_views  # noqa: E402
from util import timed  # noqa: E402

log = logging.getLogger("fit_3dgs")


# ---------------------------------------------------------------------------
# gaussian init  (primary view, all 6 depth-peel layers -- same scheme as
# fit_gsplat.py's init_gaussians, but seeding SH features instead of flat RGB)
# ---------------------------------------------------------------------------

def _logit(x, eps=1e-4):
  x = np.clip(x, eps, 1.0 - eps)
  return np.log(x / (1.0 - x))


def init_gaussians(depth_primary, K_primary, pose_primary, image_primary,
                   knn_k, init_opacity, max_sh_degree):
  """Seed one Gaussian per depth-peel hit in the primary view. Returns numpy
  arrays; `u/v/layer` record each Gaussian's pixel + peel-layer of origin (in
  depth-peel pixels). `features_dc` is the front-pixel colour in SH DC form
  (RGB2SH); `features_rest` (higher SH bands) starts at zero, matching the
  reference's `create_from_pcd`."""
  pts, u, v, layer = unproject_depth_peel(
    depth_primary, K_primary, pose_primary, space="world")          # (P,3), (P,), (P,), (P,)
  if len(pts) == 0:
    raise SystemExit("primary view has no depth-peel hits -- nothing to seed")

  dh, dw = depth_primary.shape[:2]
  ih, iw = image_primary.shape[:2]
  iv = np.clip(np.round(v * (ih / dh)), 0, ih - 1).astype(np.int64)
  iu = np.clip(np.round(u * (iw / dw)), 0, iw - 1).astype(np.int64)
  colors = image_primary[iv, iu, :3].astype(np.float32) / 255.0      # front-pixel colour

  # isotropic initial scale = mean distance to the knn_k nearest neighbours
  from scipy.spatial import cKDTree
  k = min(knn_k + 1, len(pts))
  dist, _ = cKDTree(pts).query(pts, k=k)                    # (P, k), or (P,) when k == 1
  dist = rearrange(dist, "p -> p 1") if dist.ndim == 1 else dist
  neighbours = dist[:, 1:] if k > 1 else dist               # column 0 is the point itself
  nn = reduce(neighbours, "p k -> p", "mean")
  nn = np.clip(nn, 1e-6, None).astype(np.float32)

  num_sh = (int(max_sh_degree) + 1) ** 2
  features_dc = ((colors - 0.5) / SH_C0)[:, None, :].astype(np.float32)         # (P,1,3)
  features_rest = np.zeros((len(pts), num_sh - 1, 3), np.float32)              # (P,K-1,3)

  return {
    "means": pts.astype(np.float32),
    "scales": repeat(np.log(nn), "p -> p xyz", xyz=3).astype(np.float32),
    "quats": np.tile([1.0, 0.0, 0.0, 0.0], (len(pts), 1)).astype(np.float32),
    "opacities": np.full(len(pts), _logit(np.float32(init_opacity)), np.float32),
    "features_dc": features_dc, "features_rest": features_rest,
    "u": u.astype(np.int64), "v": v.astype(np.int64), "layer": layer.astype(np.int64),
  }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(params, active_sh_degree, viewmats, Ks, width, height):
  import gsplat
  colors = torch.cat([params["features_dc"], params["features_rest"]], dim=1)  # (P,K,3)
  rgb, alpha, info = gsplat.rasterization(
    means=params["means"],
    quats=F.normalize(params["quats"], dim=-1),
    scales=torch.exp(params["scales"]),
    opacities=torch.sigmoid(params["opacities"]),
    colors=colors, sh_degree=active_sh_degree,
    viewmats=viewmats, Ks=Ks, width=width, height=height,
    render_mode="RGB", packed=True,
  )
  # no `backgrounds`: colours come back un-composited (== premultiplied by the
  # rendered alpha), which is exactly what the premultiplied GT is compared to
  # (same convention as fit_gsplat.py's render()).
  return rgb.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0), info


# ---------------------------------------------------------------------------
# camera sampling -- one-random-camera(-batch)-per-step, reshuffled every
# epoch: the reference's exact `viewpoint_stack` mechanism (pop a random
# camera, refill+reshuffle the stack when it empties), generalized to pop
# `k` at a time so `views_per_iter` > 1 also gets full per-epoch coverage --
# unlike fit_gsplat.py's views_per_iter, which draws an independent random
# subset every iteration with no such guarantee.
# ---------------------------------------------------------------------------

class _EpochQueue:
  def __init__(self, n, rng):
    self.n, self.rng = n, rng
    self.queue = np.empty(0, dtype=np.int64)

  def next(self, k):
    while len(self.queue) < k:
      self.queue = np.concatenate([self.queue, self.rng.permutation(self.n)])
    idx, self.queue = self.queue[:k], self.queue[k:]
    return idx


# ---------------------------------------------------------------------------
# opacity reset / pruning -- remove-only (see module docstring): reuses
# gsplat's own `strategy.ops` primitives, the same ones `DefaultStrategy`
# calls internally for its prune half, just invoked directly without the
# grow half (`duplicate`/`split`) since new Gaussians would land off the
# pixel-aligned seed grid.
# ---------------------------------------------------------------------------

def cameras_extent(poses):
  """poses: (V,4,4) camera-to-world. Same `getNerfppNorm` construction the
  reference uses for `scene.cameras_extent`: 1.1x the largest camera-center
  distance from the centroid of all training cameras."""
  centers = poses[:, :3, 3]
  radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1
  return radius or 1.0


def build_prune_mask(params, state, cfg, it, scene_scale):
  opac = torch.sigmoid(params["opacities"])
  mask = opac <= float(cfg.prune.min_opacity)
  if cfg.prune.get("max_scale_ratio") is not None:
    scale_max = torch.exp(params["scales"]).max(dim=-1).values
    mask = mask | (scale_max > float(cfg.prune.max_scale_ratio) * scene_scale)
  if cfg.prune.get("max_screen_size") is not None and it > int(cfg.opacity_reset.every):
    mask = mask | (state["max_radii2d"] > float(cfg.prune.max_screen_size))
  return mask


def update_max_radii2d(state, info):
  """Running per-Gaussian max of the on-screen radius (px) over every
  iteration's rendered cameras -- the reference's `max_radii2D`. Only
  Gaussians visible this iteration (`gaussian_ids`, from `packed=True`) are
  touched; everything else keeps its prior running max."""
  ids = info["gaussian_ids"]
  radii = info["radii"].max(dim=-1).values.to(state["max_radii2d"].dtype)
  update = torch.zeros_like(state["max_radii2d"]).scatter_reduce(
    0, ids, radii, reduce="amax", include_self=False)
  visible = torch.zeros_like(state["max_radii2d"], dtype=torch.bool)
  visible[ids] = True
  state["max_radii2d"] = torch.where(visible, torch.maximum(state["max_radii2d"], update), state["max_radii2d"])


# ---------------------------------------------------------------------------
# output  (same (H, W, 6, .) grid scheme as fit_gsplat.py's save_output)
# ---------------------------------------------------------------------------

def _scatter_grid(flat, v, u, layer, H, W, L):
  shape = (H, W, L) if flat.ndim == 1 else (H, W, L, flat.shape[1])
  grid = np.full(shape, np.nan, np.float32)
  grid[v, u, layer] = flat
  return grid


def _scene_dataset(f, name, data):
  f.create_dataset(name, data=data, chunks=data.shape,
                   compression="gzip", compression_opts=4)


def save_output(path, cfg, views, params_np, uvl_np, H, W, L, final_loss, scene_scale,
                active_sh_degree, num_seeded):
  v, u, layer = uvl_np["v"], uvl_np["u"], uvl_np["layer"]
  means = params_np["means"]
  scales = np.exp(params_np["scales"])
  quats = params_np["quats"] / np.linalg.norm(params_np["quats"], axis=-1, keepdims=True)
  opac = 1.0 / (1.0 + np.exp(-params_np["opacities"]))
  P, K1, _ = params_np["features_rest"].shape

  with h5py.File(path, "w") as f:
    f.attrs["config_json"] = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    f.attrs["mesh_index"] = views["mesh_index"]
    f.attrs["mesh_path"] = views["mesh_path"]
    f.attrs["final_loss"] = float(final_loss)
    f.attrs["num_gaussians"] = int(len(means))
    f.attrs["num_gaussians_seeded"] = int(num_seeded)
    f.attrs["scene_scale"] = float(scene_scale)
    f.attrs["active_sh_degree"] = int(active_sh_degree)
    f.attrs["layer_layout"] = "(H, W, 6) like depth_peel; NaN = empty slot (never seeded, or pruned)"

    _scene_dataset(f, "gaussian_means", _scatter_grid(means, v, u, layer, H, W, L))
    _scene_dataset(f, "gaussian_scales", _scatter_grid(scales, v, u, layer, H, W, L))
    _scene_dataset(f, "gaussian_quats", _scatter_grid(quats, v, u, layer, H, W, L))
    _scene_dataset(f, "gaussian_opacities", _scatter_grid(opac, v, u, layer, H, W, L))
    _scene_dataset(f, "gaussian_features_dc", _scatter_grid(params_np["features_dc"][:, 0, :], v, u, layer, H, W, L))
    frest_flat = _scatter_grid(params_np["features_rest"].reshape(P, -1), v, u, layer, H, W, L)
    _scene_dataset(f, "gaussian_features_rest", frest_flat.reshape(H, W, L, K1, 3))
    valid = np.zeros((H, W, L), bool)
    valid[v, u, layer] = True
    _scene_dataset(f, "layer_valid", valid)

    f.create_dataset("camera_pose_used", data=views["pose"])
    f.create_dataset("depth_intrinsics_used", data=views["K"])
    f.create_dataset("image_intrinsics_used", data=views["image_K"])
    _scene_dataset(f, "depth_peel_primary", views["depth"][0])
    _scene_dataset(f, "images_used", views["images"])
    f.create_dataset("view_index_used", data=np.asarray(views["order"], np.int64))


def write_ply(path, params_np, scene_scale, opacity_threshold=None,
              max_scale_ratio=None, max_anisotropy=None):
  """Standard INRIA-format 3DGS .ply with full SH (unlike fit_gsplat.py's
  write_ply, which only ever has band-0 colour). Same opt-in pruning knobs as
  fit_gsplat.py's write_ply -- see its docstring."""
  import gsplat
  opac = 1.0 / (1.0 + np.exp(-params_np["opacities"]))
  scales = np.exp(params_np["scales"])
  ax_max, ax_min = scales.max(1), np.maximum(scales.min(1), 1e-12)

  drop_opacity = (np.zeros(len(opac), bool) if opacity_threshold is None
                  else opac <= float(opacity_threshold))
  drop_huge = (np.zeros_like(drop_opacity) if max_scale_ratio is None
               else ax_max > float(max_scale_ratio) * scene_scale)
  drop_sliver = (np.zeros_like(drop_opacity) if max_anisotropy is None
                 else ax_max / ax_min > float(max_anisotropy))
  keep = ~(drop_opacity | drop_huge | drop_sliver)

  quats = params_np["quats"][keep]
  quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
  gsplat.export_splats(
    means=torch.from_numpy(params_np["means"][keep]),
    scales=torch.from_numpy(params_np["scales"][keep]),      # log-space; viewer exp()s
    quats=torch.from_numpy(quats.astype(np.float32)),
    opacities=torch.from_numpy(params_np["opacities"][keep]),  # logit; viewer sigmoid()s
    sh0=torch.from_numpy(params_np["features_dc"][keep]),
    shN=torch.from_numpy(params_np["features_rest"][keep]),
    format="ply", save_to=path,
  )
  return {
    "total": int(keep.size), "kept": int(keep.sum()),
    "low_opacity": int(drop_opacity.sum()),
    "huge": int(drop_huge.sum()), "sliver": int(drop_sliver.sum()),
  }


def _build_val_panel(gt_rgb, gt_alpha, render_rgb, render_alpha):
  V = gt_rgb.shape[0]
  rows = []
  for i in range(V):
    g = (gt_rgb[i].detach().cpu().numpy() * 255).astype(np.uint8)
    r = (render_rgb[i].detach().cpu().numpy() * 255).astype(np.uint8)
    ga = (gt_alpha[i, ..., 0].detach().cpu().numpy() * 255).astype(np.uint8)
    ra = (render_alpha[i, ..., 0].detach().cpu().numpy() * 255).astype(np.uint8)
    ga3 = np.repeat(ga[..., None], 3, axis=2)
    ra3 = np.repeat(ra[..., None], 3, axis=2)
    rows.append(np.concatenate([g, r, ga3, ra3], axis=1))
  return np.concatenate(rows, axis=0)


def dump_val(val_dir, it, panel):
  from PIL import Image
  os.makedirs(val_dir, exist_ok=True)
  Image.fromarray(panel).save(os.path.join(val_dir, f"iter{it:05d}.png"))


# ---------------------------------------------------------------------------
# orbit previews (wandb.Video, periodic during optimization) -- same
# reasoning/anchoring as fit_gsplat.py's render_orbit_frames (see its
# docstring); duplicated here (rather than imported) because it needs this
# module's SH-aware render(), not fit_gsplat.py's flat-colour one.
# ---------------------------------------------------------------------------

WORLD_UP = np.array([0.0, 0.0, 1.0], np.float32)  # Blender / render_objaverse is Z-up


def look_at_c2w(eye, target, up=WORLD_UP):
  z = eye - target
  z = z / (np.linalg.norm(z) + 1e-8)
  if abs(np.dot(z, up)) > 0.999:
    up = np.array([0.0, 1.0, 0.0], np.float32) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
  x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-8)
  y = np.cross(z, x)
  c2w = np.eye(4, dtype=np.float32)
  c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
  return c2w


def write_mp4(frames, path, fps, crf):
  import imageio.v2 as imageio
  writer = imageio.get_writer(
    path, format="FFMPEG", mode="I", fps=float(fps),
    codec="libx264", macro_block_size=1, pixelformat="yuv420p",
    ffmpeg_params=["-crf", str(int(crf))],
  )
  try:
    for fr in frames:
      writer.append_data(np.ascontiguousarray(fr))
  finally:
    writer.close()
  return path


def render_orbit_frames(params, active_sh_degree, K_ref, dist, width, height, device,
                        num_frames=24, elevation_deg=20.0):
  elev = np.radians(float(elevation_deg))
  azimuths = np.linspace(0.0, 2 * np.pi, int(num_frames), endpoint=False)
  K_t = torch.from_numpy(K_ref[None]).to(device)
  for az in azimuths:
    d = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)], np.float32)
    pose = look_at_c2w(d * dist, np.zeros(3, np.float32))
    viewmat = torch.from_numpy(make_viewmats(pose[None])).to(device)
    with torch.no_grad():
      rgb, _, _ = render(params, active_sh_degree, viewmat, K_t, width, height)
    yield (rgb[0].clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)


def log_orbit_video(wandb_run, step, frames, fps, crf, workdir):
  import wandb
  path = os.path.join(workdir, f"orbit_{step}.mp4")
  write_mp4(frames, path, fps, crf)
  wandb_run.log({"orbit": wandb.Video(path, caption=f"iter {step}", format="mp4")}, step=step)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="fit_3dgs")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(int(cfg.seed))
  device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
  if device != cfg.device:
    log.warning("cuda not available, falling back to cpu")

  out_h5 = cfg.output_path or f"{cfg.hdf5_path}.fit3dgs.view{int(cfg.primary)}.h5"
  stem = out_h5[:-3] if out_h5.endswith(".h5") else out_h5

  wandb_run = None
  if cfg.wandb.mode != "disabled":
    import wandb
    wandb_run = wandb.init(
      project=cfg.wandb.project, mode=cfg.wandb.mode, tags=list(cfg.wandb.tags),
      name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True),
    )

  with timed("load"):
    views = load_views(cfg)
  V = views["images"].shape[0]
  IH, IW = views["images"].shape[1:3]
  DH, DW, L = views["depth"].shape[1:]
  if views["images"].shape[-1] != 4 and (IH, IW) != (DH, DW):
    raise SystemExit("RGB-only file (no alpha) with images and depth_peel at "
                     "different resolutions is not supported -- the mask can't be "
                     "derived. Use an RGBA render or equal resolutions.")
  views_per_iter = cfg.get("views_per_iter", None)
  views_per_iter = None if views_per_iter is None else min(int(views_per_iter), V)
  max_sh_degree = int(cfg.max_sh_degree)
  log.info("primary=%d secondary=%s  %d views  RGB %dx%d  depth/grid %dx%d  %d peel layers  "
           "views_per_iter=%s  max_sh_degree=%d  mesh=%s",
           views["primary"], views["secondary"], V, IW, IH, DW, DH, L,
           views_per_iter if views_per_iter is not None else "all",
           max_sh_degree, views["mesh_path"] or views["mesh_index"])

  with timed("init"):
    g = init_gaussians(views["depth"][0], views["K"][0], views["pose"][0],
                       views["images"][0], int(cfg.knn_k), float(cfg.init_opacity), max_sh_degree)
  uvl = {"u": g.pop("u"), "v": g.pop("v"), "layer": g.pop("layer")}
  num_seeded = len(g["means"])
  per_layer = np.bincount(uvl["layer"], minlength=L)
  log.info("seeded %d Gaussians (from the %dx%d depth peel)  per-layer counts %s",
           num_seeded, DW, DH, per_layer.tolist())

  scene_scale = cameras_extent(views["pose"])
  params = {
    k: torch.nn.Parameter(torch.from_numpy(v).to(device), requires_grad=(k != "means"))
    for k, v in g.items()
  }
  log.info("Gaussian centers: FROZEN at depth-peel seed positions (pixel-aligned, no densification)")

  optimizers = {
    name: torch.optim.Adam([params[name]], lr=float(cfg.lr[name]))
    for name in ("features_dc", "features_rest", "opacities", "scales", "quats")
  }
  # extra per-Gaussian running state that must stay index-aligned with `params`
  # across pruning -- bundled with u/v/layer so gsplat's `remove()` (which
  # slices every torch.Tensor value in this dict by the same `sel` it uses for
  # `params`) keeps them all in sync automatically.
  state = {
    "max_radii2d": torch.zeros(num_seeded, device=device),
    "u": torch.from_numpy(uvl["u"]).to(device),
    "v": torch.from_numpy(uvl["v"]).to(device),
    "layer": torch.from_numpy(uvl["layer"]).to(device),
  }

  viewmats_all = torch.from_numpy(make_viewmats(views["pose"])).to(device)
  Ks_all = torch.from_numpy(views["image_K"]).to(device)

  gt_rgb_all = torch.from_numpy(views["images"][..., :3].astype(np.float32) / 255.0).to(device)
  if views["images"].shape[-1] == 4:
    gt_alpha_all = torch.from_numpy(views["images"][..., 3:4].astype(np.float32) / 255.0).to(device)
  else:
    gt_alpha_all = (torch.from_numpy((views["depth"][:, :, :, 0] > 0).astype(np.float32))
                    .to(device)[..., None])
  gt_rgb_all = gt_rgb_all * gt_alpha_all  # composite GT over black, matching a black-bg render

  ld = float(cfg.lambda_dssim)
  val_every = int(cfg.val_every)
  active_sh_degree = 0
  sh_degree_interval = int(cfg.sh_degree_interval)

  iters = int(cfg.iters)
  try:
    from tqdm import trange
    it_range = trange(iters, disable=None)
  except ImportError:
    it_range = range(iters)
  log_every = max(1, iters // 10)

  orbit_every = int(cfg.orbit.every) if cfg.orbit.enabled else 0
  orbit_workdir = tempfile.mkdtemp(prefix="fit_3dgs_orbit_") if orbit_every else None
  K_orbit = views["image_K"][0]

  epoch_rng = np.random.default_rng(int(cfg.seed))
  epoch_queue = _EpochQueue(V, epoch_rng) if views_per_iter is not None else None

  n_reset, n_pruned_total = 0, 0
  final_loss = float("nan")
  with timed("optimize"):
    for it in it_range:
      iteration = it + 1  # 1-indexed, matching the reference's schedule checks

      if iteration % sh_degree_interval == 0 and active_sh_degree < max_sh_degree:
        active_sh_degree += 1

      if views_per_iter is None:
        idx = None
        vm_it, ks_it, gt_rgb_it, gt_alpha_it = viewmats_all, Ks_all, gt_rgb_all, gt_alpha_all
      else:
        idx = torch.from_numpy(epoch_queue.next(views_per_iter)).to(device)
        vm_it, ks_it, gt_rgb_it, gt_alpha_it = viewmats_all[idx], Ks_all[idx], gt_rgb_all[idx], gt_alpha_all[idx]

      rgb, alpha, info = render(params, active_sh_degree, vm_it, ks_it, IW, IH)
      rgb_c = rgb * alpha
      l1 = (rgb_c - gt_rgb_it).abs().mean()
      dssim = 1.0 - ssim(rgb_c.permute(0, 3, 1, 2), gt_rgb_it.permute(0, 3, 1, 2))
      loss = (1 - ld) * l1 + ld * dssim

      for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
      loss.backward()

      if cfg.prune.enabled:
        with torch.no_grad():
          update_max_radii2d(state, info)

      for opt in optimizers.values():
        opt.step()
      final_loss = loss.item()

      with torch.no_grad():
        if cfg.opacity_reset.enabled and iteration % int(cfg.opacity_reset.every) == 0:
          from gsplat.strategy.ops import reset_opa
          reset_opa(params, optimizers, state, value=float(cfg.opacity_reset.value))
          n_reset += 1

        if (cfg.prune.enabled and int(cfg.prune.from_iter) < iteration <= int(cfg.prune.until_iter)
            and iteration % int(cfg.prune.every) == 0):
          from gsplat.strategy.ops import remove
          mask = build_prune_mask(params, state, cfg, iteration, scene_scale)
          n = int(mask.sum().item())
          if n:
            remove(params, optimizers, state, mask)
            n_pruned_total += n

      last = it == iters - 1
      n_gauss = len(params["means"])
      if hasattr(it_range, "set_postfix") and it % 10 == 0:
        it_range.set_postfix(loss=final_loss, l1=l1.item(), dssim=dssim.item(),
                             sh=active_sh_degree, n=n_gauss)
      if it % log_every == 0 or last:
        log.info("iter %d/%d  loss %.5f  (l1 %.5f  dssim %.5f)  sh_degree=%d  "
                 "gaussians=%d  pruned_total=%d  opacity_resets=%d",
                 it, iters, final_loss, l1.item(), dssim.item(), active_sh_degree,
                 n_gauss, n_pruned_total, n_reset)
      if wandb_run is not None and (it % 10 == 0 or last):
        wandb_run.log({
          "train/loss": final_loss, "train/l1": l1.item(), "train/dssim": dssim.item(),
          "train/active_sh_degree": active_sh_degree, "train/num_gaussians": n_gauss,
        }, step=it)
      if val_every and (it % val_every == 0 or last):
        if idx is None:
          full_rgb_c, full_alpha = rgb_c, alpha
        else:
          with torch.no_grad():
            full_rgb, full_alpha, _ = render(params, active_sh_degree, viewmats_all, Ks_all, IW, IH)
            full_rgb_c = full_rgb * full_alpha
        panel = _build_val_panel(gt_rgb_all, gt_alpha_all, full_rgb_c, full_alpha)
        dump_val(f"{stem}.val", it, panel)
        if wandb_run is not None:
          import wandb
          wandb_run.log({"val/panel": wandb.Image(panel, caption=f"iter {it}")}, step=it)
      if orbit_every and wandb_run is not None and (it % orbit_every == 0 or last):
        frames = render_orbit_frames(
          params, active_sh_degree, K_orbit, scene_scale, IW, IH, device,
          num_frames=cfg.orbit.num_frames, elevation_deg=cfg.orbit.elevation_deg,
        )
        log_orbit_video(wandb_run, it, frames, cfg.orbit.fps, cfg.orbit.crf, orbit_workdir)

  params_np = {k: v.detach().cpu().numpy() for k, v in params.items()}
  uvl_np = {k: state[k].detach().cpu().numpy() for k in ("u", "v", "layer")}
  with timed("write"):
    save_output(out_h5, cfg, views, params_np, uvl_np, DH, DW, L, final_loss, scene_scale,
               active_sh_degree, num_seeded)
    if bool(cfg.write_ply):
      st = write_ply(f"{stem}.ply", params_np, scene_scale,
                     cfg.get("ply_opacity_threshold"),
                     cfg.get("ply_max_scale_ratio"), cfg.get("ply_max_anisotropy"))
      log.info(".ply: kept %d/%d Gaussians  (pruned %d low-opacity, %d huge, %d sliver)",
               st["kept"], st["total"], st["low_opacity"], st["huge"], st["sliver"])

  log.info("final loss %.5f  %d/%d Gaussians survived (%d pruned, %d opacity resets)  ->  %s%s%s",
           final_loss, len(params_np["means"]), num_seeded, n_pruned_total, n_reset, out_h5,
           f"  {stem}.ply" if cfg.write_ply else "",
           f"  {stem}.val/" if val_every else "")
  if wandb_run is not None:
    wandb_run.summary["final_loss"] = final_loss
    wandb_run.summary["num_gaussians"] = len(params_np["means"])
    wandb_run.summary["num_gaussians_seeded"] = num_seeded
    wandb_run.finish()
  if orbit_workdir is not None:
    shutil.rmtree(orbit_workdir, ignore_errors=True)


if __name__ == "__main__":
  main()
