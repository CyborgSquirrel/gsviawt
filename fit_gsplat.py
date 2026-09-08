#!/usr/bin/env python3
"""Optimize a layered 3D Gaussian Splatting model from one render HDF5.

Input is an `.h5` in the `render_objaverse.py` / `capture_turntable.py` schema
(`images`, `depth_peel`, `camera_intrinsics`, `camera_pose`, `mesh_index`). You
pick one **primary** view index and zero or more **secondary** view indices; all
must show the same object (same `mesh_index`).

The Gaussians are seeded entirely from the primary view's 6-layer depth peel --
every pixel-hit in every peel layer becomes one Gaussian, back-projected to world
space with that view's camera (reusing `debug_pointcloud.unproject_depth_peel`).
Each Gaussian therefore has a well-defined origin `(v, u, layer)` in the primary
depth map, and the optimizer never adds or removes Gaussians (no densification),
so that 1:1 correspondence survives to the output.

They are then optimized with `gsplat` against the RGB (L1 + D-SSIM) and alpha
(L1) of the primary + secondary views. Secondary views contribute supervision
only, never new Gaussians. Fully-occluded deeper-layer Gaussians (seen in no
supplied view) keep their initial front-pixel colour.

By default (`optimize_means: false`) the Gaussian centers are locked to those
back-projected seed positions and only scale / rotation / opacity / colour are
optimized, so `gaussian_means` in the output equals the unprojected depth peel
exactly. Set `optimize_means: true` to let the centers move too.

Output is written next to the input as:
  <h5>.gsplat.view<primary>.h5  -- Gaussian attributes as (H, W, 6, .) grids
      (NaN marks an empty slot, same layout as `depth_peel` / wt_infer_layers'
      `points`), plus flat `*_flat` copies and the primary/secondary cameras.
  <h5>.gsplat.view<primary>.ply  -- standard 3DGS point cloud (if write_ply)
  <h5>.gsplat.view<primary>.val/ -- gt|render comparison PNGs (if val_every)

Config is Hydra (`conf/gsplat.yaml`); run in the container venv, e.g.

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python fit_gsplat.py \\
        hdf5_path=/app/bla/obj_lite.h5 primary=0 'secondary=[1,2,3]' iters=1500

gsplat JIT-compiles CUDA kernels on first import; this module points it at the
pip `nvidia-cuda-nvcc` toolchain (the base image has no system `nvcc`).
"""

import logging
import os
import sys
import sysconfig


def _setup_cuda_toolchain() -> None:
  """Make the pip-installed CUDA toolkit (`nvidia-cuda-nvcc` etc.) usable by
  gsplat's `torch.utils.cpp_extension` JIT build: export CUDA_HOME / PATH and
  add the `libcudart.so` dev symlink the linker's `-lcudart` needs (the wheel
  ships only `libcudart.so.13`). No-ops cleanly if the layout isn't there or a
  system CUDA is already configured."""
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_pointcloud import unproject_depth_peel  # noqa: E402
from util import timed  # noqa: E402

log = logging.getLogger("fit_gsplat")

# camera-local axis flip: Blender/OpenGL (X right, Y up, Z back) <-> OpenCV
# (X right, Y down, Z forward). Same flip `debug_pointcloud` applies inline.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
SH_C0 = 0.28209479177387814  # SH band-0 constant, for RGB <-> sh0 in the .ply


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_views(cfg: DictConfig):
  """Read the primary + secondary views. Returns a dict of stacked arrays
  (primary first) plus the raw primary-view arrays used for seeding."""
  ds = cfg.datasets
  primary = int(cfg.primary)
  secondary = [int(i) for i in cfg.secondary if int(i) != primary]
  seen = set()
  secondary = [i for i in secondary if not (i in seen or seen.add(i))]
  order = [primary, *secondary]

  with h5py.File(cfg.hdf5_path, "r") as f:
    n = f[ds.pose].shape[0]
    for i in order:
      if not 0 <= i < n:
        raise SystemExit(f"view index {i} out of range [0, {n}) in {cfg.hdf5_path}")

    if "mesh_index" in f:
      mi = f["mesh_index"][:]
      picked = {int(mi[i]) for i in order}
      if len(picked) > 1:
        raise SystemExit(
          f"selected views span mesh_index {sorted(picked)} -- primary + secondary "
          f"views must all be the same object")
      mesh_idx = int(mi[primary])
    else:
      mesh_idx = -1

    mesh_path = ""
    if "mesh_paths" in f and 0 <= mesh_idx < f["mesh_paths"].shape[0]:
      mp = f["mesh_paths"][mesh_idx]
      mesh_path = mp.decode() if isinstance(mp, bytes) else str(mp)

    images = np.stack([np.asarray(f[ds.images][i]) for i in order])          # (V,H,W,3|4) u8
    depth = np.stack([np.asarray(f[ds.depth][i]) for i in order]).astype(np.float32)  # (V,H,W,L)
    K = np.stack([np.asarray(f[ds.intrinsics][i]) for i in order]).astype(np.float32)  # (V,3,3)
    pose = np.stack([np.asarray(f[ds.pose][i]) for i in order]).astype(np.float32)     # (V,4,4) c2w

  return {
    "order": order, "primary": primary, "secondary": secondary,
    "mesh_index": mesh_idx, "mesh_path": mesh_path,
    "images": images, "depth": depth, "K": K, "pose": pose,
  }


# ---------------------------------------------------------------------------
# gaussian init  (primary view, all 6 depth-peel layers)
# ---------------------------------------------------------------------------

def _logit(x, eps=1e-4):
  x = np.clip(x, eps, 1.0 - eps)
  return np.log(x / (1.0 - x))


def init_gaussians(depth_primary, K_primary, pose_primary, image_primary,
                   knn_k, init_opacity):
  """Seed one Gaussian per depth-peel hit in the primary view. Returns numpy
  arrays; `u/v/layer` record each Gaussian's pixel + peel-layer of origin."""
  pts, u, v, layer = unproject_depth_peel(
    depth_primary, K_primary, pose_primary, space="world")          # (P,3), (P,), (P,), (P,)
  if len(pts) == 0:
    raise SystemExit("primary view has no depth-peel hits -- nothing to seed")

  colors = image_primary[v, u, :3].astype(np.float32) / 255.0        # front-pixel colour

  # isotropic initial scale = mean distance to the knn_k nearest neighbours
  from scipy.spatial import cKDTree
  k = min(knn_k + 1, len(pts))
  dist, _ = cKDTree(pts).query(pts, k=k)
  dist = np.atleast_2d(dist.T).T
  nn = dist[:, 1:].mean(axis=1) if dist.shape[1] > 1 else dist[:, 0]
  nn = np.clip(nn, 1e-6, None).astype(np.float32)

  return {
    "means": pts.astype(np.float32),
    "scales_log": np.log(nn)[:, None].repeat(3, axis=1),
    "quats": np.tile([1.0, 0.0, 0.0, 0.0], (len(pts), 1)).astype(np.float32),
    "opac_logit": np.full(len(pts), _logit(np.float32(init_opacity)), np.float32),
    "colors_logit": _logit(colors).astype(np.float32),
    "u": u.astype(np.int64), "v": v.astype(np.int64), "layer": layer.astype(np.int64),
  }


# ---------------------------------------------------------------------------
# rendering / losses
# ---------------------------------------------------------------------------

def make_viewmats(poses):
  """poses: (V,4,4) camera-to-world, OpenGL cam axes. Returns (V,4,4)
  world-to-camera in OpenCV convention -- what gsplat wants."""
  c2w_cv = poses @ OPENGL_TO_OPENCV
  return np.linalg.inv(c2w_cv).astype(np.float32)


def _gaussian_window(size=11, sigma=1.5, device="cpu"):
  coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
  g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
  g = (g / g.sum())
  return (g[:, None] * g[None, :])[None, None]


def ssim(x, y, window=None):
  """x, y: (V,3,H,W) in [0,1]. Mean SSIM over the batch (standard 11x11
  Gaussian-window SSIM, channels filtered independently)."""
  if window is None:
    window = _gaussian_window(device=x.device)
  c = x.shape[1]
  w = window.expand(c, 1, -1, -1)
  pad = w.shape[-1] // 2
  mu_x = F.conv2d(x, w, padding=pad, groups=c)
  mu_y = F.conv2d(y, w, padding=pad, groups=c)
  mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
  sig_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
  sig_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
  sig_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
  c1, c2 = 0.01 ** 2, 0.03 ** 2
  s = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
  return s.mean()


def render(params, viewmats, Ks, width, height):
  import gsplat
  rgb, alpha, _ = gsplat.rasterization(
    means=params["means"],
    quats=F.normalize(params["quats"], dim=-1),
    scales=torch.exp(params["scales_log"]),
    opacities=torch.sigmoid(params["opac_logit"]),
    colors=torch.sigmoid(params["colors_logit"]),
    viewmats=viewmats, Ks=Ks, width=width, height=height,
    sh_degree=None, render_mode="RGB", packed=True,
  )
  # no `backgrounds`: colours come back un-composited (== premultiplied by the
  # rendered alpha), which is exactly what the premultiplied GT is compared to.
  return rgb.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _scatter_grid(flat, v, u, layer, H, W, L):
  """flat: (G,) or (G,C). Returns (H,W,L) or (H,W,L,C) float32 with NaN in
  every slot that has no Gaussian."""
  shape = (H, W, L) if flat.ndim == 1 else (H, W, L, flat.shape[1])
  grid = np.full(shape, np.nan, np.float32)
  grid[v, u, layer] = flat
  return grid


def save_output(path, cfg, views, params_np, uvl, H, W, L, final_loss, scene_scale):
  v, u, layer = uvl["v"], uvl["u"], uvl["layer"]
  means = params_np["means"]
  scales = np.exp(params_np["scales_log"])
  quats = params_np["quats"] / np.linalg.norm(params_np["quats"], axis=-1, keepdims=True)
  opac = 1.0 / (1.0 + np.exp(-params_np["opac_logit"]))
  colors = 1.0 / (1.0 + np.exp(-params_np["colors_logit"]))

  with h5py.File(path, "w") as f:
    f.attrs["config_json"] = OmegaConf.to_yaml(cfg)
    f.attrs["source_h5"] = str(cfg.hdf5_path)
    f.attrs["primary_index"] = views["primary"]
    f.attrs["secondary_indices"] = np.asarray(views["secondary"], np.int64)
    f.attrs["mesh_index"] = views["mesh_index"]
    f.attrs["mesh_path"] = views["mesh_path"]
    f.attrs["iters"] = int(cfg.iters)
    f.attrs["optimize_means"] = bool(cfg.optimize_means)
    f.attrs["final_loss"] = float(final_loss)
    f.attrs["num_gaussians"] = int(len(means))
    f.attrs["scene_scale"] = float(scene_scale)
    f.attrs["layer_layout"] = "(H, W, 6) like depth_peel; NaN = empty slot"

    # per-layer Gaussian grids (H, W, L, .)
    f.create_dataset("gaussian_means", data=_scatter_grid(means, v, u, layer, H, W, L))
    f.create_dataset("gaussian_scales", data=_scatter_grid(scales, v, u, layer, H, W, L))
    f.create_dataset("gaussian_quats", data=_scatter_grid(quats, v, u, layer, H, W, L))
    f.create_dataset("gaussian_opacities", data=_scatter_grid(opac, v, u, layer, H, W, L))
    f.create_dataset("gaussian_colors", data=_scatter_grid(colors, v, u, layer, H, W, L))
    valid = np.zeros((H, W, L), bool)
    valid[v, u, layer] = True
    f.create_dataset("layer_valid", data=valid)

    # flat copies (Gaussian order == init order == unproject_depth_peel order)
    f.create_dataset("means_flat", data=means)
    f.create_dataset("scales_flat", data=scales.astype(np.float32))
    f.create_dataset("quats_flat", data=quats.astype(np.float32))
    f.create_dataset("opacities_flat", data=opac.astype(np.float32))
    f.create_dataset("colors_flat", data=colors.astype(np.float32))
    f.create_dataset("layer_index_flat", data=layer)
    f.create_dataset("uv_flat", data=np.stack([u, v], axis=1))

    # cameras / GT for the views actually used (primary first)
    f.create_dataset("camera_pose_used", data=views["pose"])
    f.create_dataset("camera_intrinsics_used", data=views["K"])
    f.create_dataset("depth_peel_primary", data=views["depth"][0])
    f.create_dataset("images_used", data=views["images"])
    f.create_dataset("view_index_used", data=np.asarray(views["order"], np.int64))


def write_ply(path, params_np, scene_scale, opacity_threshold=None,
              max_scale_ratio=None, max_anisotropy=None):
  """Standard INRIA-format 3DGS .ply. `gsplat.export_splats` writes every
  field raw, and viewers apply the activations themselves: exp(scale),
  sigmoid(opacity), SH_C0*f_dc + 0.5. So pass the *unactivated* optimizer
  params -- log-scales, logit-opacities, SH-DC colours -- not the activated
  values stored in the .h5.

  All pruning is opt-in (None = off, the default), by:
    - low opacity:  sigmoid(opacity) <= opacity_threshold
    - huge: largest scale axis  >  max_scale_ratio * scene_scale
    - sliver: scale max/min ratio  >  max_anisotropy
  export_splats' own opacity threshold only applies to the compressed format,
  not plain "ply". Returns a {reason: count} dict plus "kept"/"total"."""
  import gsplat
  opac = 1.0 / (1.0 + np.exp(-params_np["opac_logit"]))
  scales = np.exp(params_np["scales_log"])                 # (N,3) world units
  ax_max, ax_min = scales.max(1), np.maximum(scales.min(1), 1e-12)

  drop_opacity = (np.zeros(len(opac), bool) if opacity_threshold is None
                  else opac <= float(opacity_threshold))
  drop_huge = (np.zeros_like(drop_opacity) if max_scale_ratio is None
               else ax_max > float(max_scale_ratio) * scene_scale)
  drop_sliver = (np.zeros_like(drop_opacity) if max_anisotropy is None
                 else ax_max / ax_min > float(max_anisotropy))
  keep = ~(drop_opacity | drop_huge | drop_sliver)

  colors = (1.0 / (1.0 + np.exp(-params_np["colors_logit"])))[keep]
  sh0 = ((colors - 0.5) / SH_C0)[:, None, :]           # (N,1,3) SH band-0 coeff
  quats = params_np["quats"][keep]
  quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
  gsplat.export_splats(
    means=torch.from_numpy(params_np["means"][keep]),
    scales=torch.from_numpy(params_np["scales_log"][keep]),  # log-space; viewer exp()s
    quats=torch.from_numpy(quats.astype(np.float32)),
    opacities=torch.from_numpy(params_np["opac_logit"][keep]),  # logit; viewer sigmoid()s
    sh0=torch.from_numpy(sh0.astype(np.float32)),
    shN=torch.zeros(len(colors), 0, 3),
    format="ply", save_to=path,
  )
  return {
    "total": int(keep.size), "kept": int(keep.sum()),
    "low_opacity": int(drop_opacity.sum()),
    "huge": int(drop_huge.sum()), "sliver": int(drop_sliver.sum()),
  }


def dump_val(val_dir, it, gt_rgb, gt_alpha, render_rgb, render_alpha):
  from PIL import Image
  os.makedirs(val_dir, exist_ok=True)
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
  Image.fromarray(np.concatenate(rows, axis=0)).save(
    os.path.join(val_dir, f"iter{it:05d}.png"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="gsplat")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(int(cfg.seed))
  device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
  if device != cfg.device:
    log.warning("cuda not available, falling back to cpu")

  out_h5 = cfg.output_path or f"{cfg.hdf5_path}.gsplat.view{int(cfg.primary)}.h5"
  stem = out_h5[:-3] if out_h5.endswith(".h5") else out_h5

  with timed("load"):
    views = load_views(cfg)
  V, H, W, L = views["images"].shape[0], *views["depth"].shape[1:]
  log.info("primary=%d secondary=%s  %d views  %dx%d  %d peel layers  mesh=%s",
           views["primary"], views["secondary"], V, W, H, L, views["mesh_path"] or views["mesh_index"])

  with timed("init"):
    g = init_gaussians(views["depth"][0], views["K"][0], views["pose"][0],
                       views["images"][0], int(cfg.knn_k), float(cfg.init_opacity))
  uvl = {"u": g.pop("u"), "v": g.pop("v"), "layer": g.pop("layer")}
  n_gauss = len(g["means"])
  per_layer = np.bincount(uvl["layer"], minlength=L)
  log.info("seeded %d Gaussians  per-layer counts %s", n_gauss, per_layer.tolist())

  scene_scale = float(np.linalg.norm(views["pose"][0][:3, 3])) or 1.0
  optimize_means = bool(cfg.optimize_means)
  params = {
    k: torch.nn.Parameter(torch.from_numpy(v).to(device),
                          requires_grad=(k != "means" or optimize_means))
    for k, v in g.items()
  }
  log.info("Gaussian centers: %s",
           "trainable" if optimize_means else "FROZEN at depth-peel seed positions")

  viewmats = torch.from_numpy(make_viewmats(views["pose"])).to(device)
  Ks = torch.from_numpy(views["K"]).to(device)

  gt_rgb = torch.from_numpy(views["images"][..., :3].astype(np.float32) / 255.0).to(device)
  if views["images"].shape[-1] == 4:
    gt_alpha = torch.from_numpy(views["images"][..., 3:4].astype(np.float32) / 255.0).to(device)
  else:
    gt_alpha = (torch.from_numpy((views["depth"][:, :, :, 0] >= 0).astype(np.float32))
                .to(device)[..., None])
  gt_rgb = gt_rgb * gt_alpha  # composite GT over black, matching a black-bg render

  groups = [
    {"params": [params["scales_log"]], "lr": float(cfg.lr.scales)},
    {"params": [params["quats"]], "lr": float(cfg.lr.quats)},
    {"params": [params["opac_logit"]], "lr": float(cfg.lr.opacities)},
    {"params": [params["colors_logit"]], "lr": float(cfg.lr.colors)},
  ]
  if optimize_means:
    groups.insert(0, {"params": [params["means"]], "lr": float(cfg.lr.means) * scene_scale})
  opt = torch.optim.Adam(groups)
  window = _gaussian_window(device=device)
  ls, lm = float(cfg.lambda_ssim), float(cfg.lambda_mask)
  val_every = int(cfg.val_every)

  iters = int(cfg.iters)
  try:
    from tqdm import trange
    it_range = trange(iters, disable=None)  # disable=None -> auto-off when not a tty
  except ImportError:
    it_range = range(iters)
  log_every = max(1, iters // 10)

  final_loss = float("nan")
  with timed("optimize"):
    for it in it_range:
      rgb, alpha = render(params, viewmats, Ks, W, H)
      rgb_c = rgb * alpha  # premultiply so bg stays black on both sides
      l1 = (rgb_c - gt_rgb).abs().mean()
      dssim = 1.0 - ssim(rgb_c.permute(0, 3, 1, 2), gt_rgb.permute(0, 3, 1, 2), window)
      mask = (alpha - gt_alpha).abs().mean()
      loss = (1 - ls) * l1 + ls * dssim + lm * mask

      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      final_loss = loss.item()

      last = it == iters - 1
      if hasattr(it_range, "set_postfix") and it % 10 == 0:
        it_range.set_postfix(loss=final_loss, l1=l1.item(), dssim=dssim.item(), mask=mask.item())
      if it % log_every == 0 or last:
        log.info("iter %d/%d  loss %.5f  (l1 %.5f  dssim %.5f  mask %.5f)",
                 it, iters, final_loss, l1.item(), dssim.item(), mask.item())
      if val_every and (it % val_every == 0 or last):
        dump_val(f"{stem}.val", it, gt_rgb, gt_alpha, rgb_c, alpha)

  params_np = {k: v.detach().cpu().numpy() for k, v in params.items()}
  with timed("write"):
    save_output(out_h5, cfg, views, params_np, uvl, H, W, L, final_loss, scene_scale)
    if bool(cfg.write_ply):
      st = write_ply(f"{stem}.ply", params_np, scene_scale,
                     cfg.ply_opacity_threshold,
                     cfg.get("ply_max_scale_ratio"), cfg.get("ply_max_anisotropy"))
      log.info(".ply: kept %d/%d Gaussians  (pruned %d low-opacity, %d huge, %d sliver)",
               st["kept"], st["total"], st["low_opacity"], st["huge"], st["sliver"])

  log.info("final loss %.5f  ->  %s%s%s", final_loss, out_h5,
           f"  {stem}.ply" if cfg.write_ply else "",
           f"  {stem}.val/" if val_every else "")


if __name__ == "__main__":
  main()
