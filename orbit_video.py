#!/usr/bin/env python3
"""Render a showcase turntable video of a 3D Gaussian Splatting model.

Takes one 3DGS model of a single object and flies the camera in a circle
around it, aimed at the object, writing an H.264 .mp4 (or, with
`format: frames`, a folder of PNGs).

Two input formats, picked from the file extension:

  *.h5   -- a `fit_gsplat.py` output: `gaussian_means` / `gaussian_scales` /
            `gaussian_quats` / `gaussian_opacities` / `gaussian_colors` as
            (H, W, 6, .) grids plus a `layer_valid` mask. Values are already
            activated (world-space means, exp'd scales, sigmoid'd opacity /
            colour, unit quats).
  *.ply  -- a standard INRIA-format 3DGS point cloud (what `fit_gsplat.py`
            also writes, and what most 3DGS viewers eat). Fields are stored
            un-activated: colour = SH_C0 * f_dc + 0.5, opacity = sigmoid,
            scale = exp, rot = normalised quat.

The object from `fit_gsplat` lives near the origin at roughly unit-cube size
(the renderer normalises it there), so the default orbit circles the origin.
`look_at: centroid` / an explicit `camera_distance` cover models that don't.

Rendering uses `gsplat.rasterization` with the same OpenGL->OpenCV camera
convention as `fit_gsplat.make_viewmats`. Config is Hydra (`conf/orbit.yaml`);
run in the container venv, e.g.

    docker exec -w /app/.claude/worktrees/orbit-video gsviawt-app-gpu-1 \\
        /home/user/venv/bin/python orbit_video.py \\
        model_path=/app/bla/obj_lite.h5.gsplat.view0.ply seconds=8 elevation_deg=25

The .mp4 is encoded through `imageio` / `imageio-ffmpeg`, which ships its own
static ffmpeg -- nothing is needed on the system PATH.

Mesh comparison
---------------
Set `mesh_path` (or leave it `auto` for a .h5 model, which records its source
mesh) to also render that mesh along the *same* orbit and emit a side-by-side
[mesh | 3DGS | abs-diff] video plus a per-frame PSNR / SSIM / MAE CSV instead
of the plain showcase. The mesh render shells out to Blender running
`orbit_render_mesh.py` (render_objaverse.py's normalise + multi-sun scene, the
distribution the 3DGS was fit against), so the two renders see pixel-identical
cameras. Cheap next to the splat pass -- a Cycles frame per pose -- so keep the
frame count modest for a first look.
"""

import csv
import logging
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# fit_gsplat wires up the pip CUDA toolchain for gsplat's JIT build on import,
# and gives us the camera-convention helper, SSIM, and the SH constant.
from fit_gsplat import make_viewmats, ssim, SH_C0  # noqa: E402

import h5py  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from util import timed  # noqa: E402

log = logging.getLogger("orbit_video")

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD_UP = np.array([0.0, 0.0, 1.0], np.float32)  # Blender / render_objaverse is Z-up


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def load_h5(path, ds):
  """fit_gsplat .h5 -> dict of flat, already-activated Gaussian arrays."""
  with h5py.File(path, "r") as f:
    valid = np.asarray(f[ds.valid][:]).astype(bool)          # (H, W, L)
    def flat(name, c):
      g = np.asarray(f[ds[name]][:]).astype(np.float32)      # (H, W, L) or (H, W, L, c)
      return g[valid].reshape(-1, c) if c > 1 else g[valid].reshape(-1)
    out = {
      "means": flat("means", 3),
      "scales": flat("scales", 3),
      "quats": flat("quats", 4),
      "opacities": flat("opacities", 1),
      "colors": flat("colors", 3),
    }
  return out


def _read_ply(path):
  """Minimal reader for the one 'vertex' element of an INRIA 3DGS .ply
  (binary_little_endian or ascii). Returns {property_name: (N,) float32}."""
  with open(path, "rb") as fh:
    if fh.readline().strip() != b"ply":
      raise SystemExit(f"{path}: not a .ply file")
    fmt = None
    count = None
    names = []
    _np_of = {
      "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
      "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
      "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
      "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
    }
    dtype = []
    while True:
      line = fh.readline()
      if not line:
        raise SystemExit(f"{path}: unexpected EOF in header")
      tok = line.split()
      if tok[0] == b"format":
        fmt = tok[1].decode()
      elif tok[0] == b"element" and tok[1] == b"vertex":
        count = int(tok[2])
      elif tok[0] == b"element":
        raise SystemExit(f"{path}: unsupported extra element {tok[1].decode()!r}")
      elif tok[0] == b"property":
        name = tok[2].decode()
        names.append(name)
        dtype.append((name, _np_of[tok[1].decode()]))
      elif tok[0] == b"end_header":
        break

    if fmt == "binary_little_endian":
      arr = np.frombuffer(fh.read(np.dtype(dtype).itemsize * count),
                          dtype=np.dtype(dtype), count=count)
      cols = {n: arr[n].astype(np.float32) for n in names}
    elif fmt == "ascii":
      raw = np.loadtxt(fh, dtype=np.float32, max_rows=count).reshape(count, len(names))
      cols = {n: raw[:, i] for i, n in enumerate(names)}
    else:
      raise SystemExit(f"{path}: unsupported ply format {fmt!r}")
  return cols


def load_ply(path):
  """Standard 3DGS .ply -> dict of flat, activated Gaussian arrays."""
  c = _read_ply(path)
  need = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
          "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
  missing = [k for k in need if k not in c]
  if missing:
    raise SystemExit(f"{path}: missing 3DGS properties {missing}")
  means = np.stack([c["x"], c["y"], c["z"]], -1)
  scales = np.exp(np.stack([c["scale_0"], c["scale_1"], c["scale_2"]], -1))
  quats = np.stack([c["rot_0"], c["rot_1"], c["rot_2"], c["rot_3"]], -1)  # wxyz
  quats = quats / np.clip(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-12, None)
  opac = _sigmoid(c["opacity"])
  colors = SH_C0 * np.stack([c["f_dc_0"], c["f_dc_1"], c["f_dc_2"]], -1) + 0.5
  return {
    "means": means.astype(np.float32),
    "scales": scales.astype(np.float32),
    "quats": quats.astype(np.float32),
    "opacities": opac.astype(np.float32),
    "colors": np.clip(colors, 0.0, 1.0).astype(np.float32),
  }


def load_model(cfg):
  ext = os.path.splitext(cfg.model_path)[1].lower()
  if ext in (".h5", ".hdf5"):
    g = load_h5(cfg.model_path, cfg.datasets)
  elif ext == ".ply":
    g = load_ply(cfg.model_path)
  else:
    raise SystemExit(f"unsupported model extension {ext!r} (want .h5 or .ply)")
  finite = np.isfinite(g["means"]).all(1)
  for k in g:
    g[k] = g[k][finite]
  if len(g["means"]) == 0:
    raise SystemExit("model has no valid Gaussians")
  return g


def resolve_mesh_path(cfg):
  """The mesh to compare against, or None for the plain showcase. `auto` reads
  a .h5 model's `mesh_path` attribute (fit_gsplat records it); an explicit
  path is taken as-is."""
  mp = cfg.get("mesh_path")
  if mp in (None, "null", ""):
    return None
  if str(mp) != "auto":
    if not os.path.exists(mp):
      raise SystemExit(f"mesh_path {mp} does not exist")
    return str(mp)
  if os.path.splitext(cfg.model_path)[1].lower() not in (".h5", ".hdf5"):
    log.info("mesh_path=auto but model is not a .h5 (no recorded mesh) -- plain showcase")
    return None
  with h5py.File(cfg.model_path, "r") as f:
    rec = f.attrs.get("mesh_path", "")
  rec = rec.decode() if isinstance(rec, bytes) else str(rec)
  if not rec:
    log.info("mesh_path=auto but the .h5 records no mesh_path -- plain showcase")
    return None
  if not os.path.exists(rec):
    raise SystemExit(f"the .h5 records mesh_path {rec!r} but it does not exist here "
                     f"(pass mesh_path=/path/to/mesh explicitly)")
  return rec


# ---------------------------------------------------------------------------
# camera path
# ---------------------------------------------------------------------------

def look_at_c2w(eye, target):
  """OpenGL camera-to-world (X right, Y up, -Z forward) looking from `eye` at
  `target`, world up = +Z. Matches render_objaverse's `to_track_quat('-Z','Y')`."""
  z = eye - target                                        # camera +Z points back
  z /= np.linalg.norm(z)
  up = WORLD_UP if abs(np.dot(z, WORLD_UP)) < 0.999 else np.array([0.0, 1.0, 0.0], np.float32)
  x = np.cross(up, z); x /= np.linalg.norm(x)
  y = np.cross(z, x)
  c2w = np.eye(4, dtype=np.float32)
  c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
  return c2w


def orbit_poses(cfg, means):
  n = int(cfg.num_frames) if cfg.num_frames else max(1, round(float(cfg.seconds) * float(cfg.fps)))

  center = means.mean(0) if str(cfg.look_at) == "centroid" else np.zeros(3, np.float32)

  fx = float(cfg.width) * float(cfg.camera_lens) / float(cfg.camera_sensor_width)

  if cfg.camera_distance is not None:
    dist = float(cfg.camera_distance)
  else:
    # bounding-sphere radius (99th pct trims stray splats), fit into the
    # narrower of the two half-FOVs with `margin` headroom.
    r = float(np.quantile(np.linalg.norm(means - center, axis=1), 0.99))
    half_fov = np.arctan(min(float(cfg.width), float(cfg.height)) / (2.0 * fx))
    dist = r / np.sin(half_fov) * float(cfg.margin)

  elev = np.radians(np.clip(float(cfg.elevation_deg), -89.0, 89.0))
  sweep = np.radians(float(cfg.orbit_deg)) * (-1.0 if cfg.clockwise else 1.0)
  a0 = np.radians(float(cfg.start_azimuth_deg))
  # endpoint=False over a full 360 so the loop doesn't stutter on a repeat frame
  full = abs(float(cfg.orbit_deg) % 360.0) < 1e-6 and float(cfg.orbit_deg) != 0.0
  fracs = np.linspace(0.0, 1.0, n, endpoint=not full)

  poses = []
  for fr in fracs:
    az = a0 + sweep * fr
    d = np.array([np.cos(elev) * np.cos(az),
                  np.cos(elev) * np.sin(az),
                  np.sin(elev)], np.float32)
    poses.append(look_at_c2w(center + d * dist, center))
  K = np.array([[fx, 0.0, float(cfg.width) / 2.0],
                [0.0, fx, float(cfg.height) / 2.0],
                [0.0, 0.0, 1.0]], np.float32)
  return np.stack(poses), K, dist, center, n


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_frames(cfg, g, poses, K, device, out_npy=None):
  """Rasterise the splat at every pose, each frame composited over
  cfg.background -> (H,W,3) uint8. Returns a list, or -- if `out_npy` is given
  -- streams the frames into that (N,H,W,3) uint8 .npy and returns its path
  (so a long comparison run never holds the whole orbit in RAM)."""
  import gsplat

  t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in g.items()}
  quats = F.normalize(t["quats"], dim=-1)
  bg = torch.tensor(list(cfg.background), dtype=torch.float32, device=device)

  Ks = torch.from_numpy(K).to(device)[None]
  W, H = int(cfg.width), int(cfg.height)
  sink = (np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8,
                                    shape=(len(poses), H, W, 3))
          if out_npy else None)
  frames = [] if sink is None else None
  for i in range(len(poses)):
    viewmat = torch.from_numpy(make_viewmats(poses[i][None])).to(device)
    rgb, alpha, _ = gsplat.rasterization(
      means=t["means"], quats=quats, scales=t["scales"],
      opacities=t["opacities"], colors=t["colors"],
      viewmats=viewmat, Ks=Ks, width=W, height=H,
      sh_degree=None, render_mode="RGB", packed=True,
      near_plane=float(cfg.near_plane), far_plane=float(cfg.far_plane),
    )
    # colours come back premultiplied by alpha; composite over the bg colour.
    comp = rgb[0].clamp(0.0, 1.0) + (1.0 - alpha[0].clamp(0.0, 1.0)) * bg
    img = (comp.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
    if sink is None:
      frames.append(img)
    else:
      sink[i] = img
    if (i + 1) % max(1, len(poses) // 10) == 0 or i == len(poses) - 1:
      log.info("rendered %d/%d frames", i + 1, len(poses))
  if sink is not None:
    sink.flush()
    del sink
    return out_npy
  return frames


# ---------------------------------------------------------------------------
# mesh render (Blender subprocess) + comparison
# ---------------------------------------------------------------------------

def render_mesh_frames(cfg, poses, K, mesh_path, workdir):
  """Render `mesh_path` along `poses` via Blender / orbit_render_mesh.py.
  Returns the path to an (N, H, W, 4) uint8 .npy (RGBA, premultiplied-alpha
  film) -- kept on disk and mmap'd by the comparison, so the frames never all
  sit in RAM at once."""
  job = os.path.join(workdir, "job.npz")
  out = os.path.join(workdir, "mesh_frames.npy")
  m = cfg.mesh
  np.savez(job, poses=poses.astype(np.float64), fx=np.float64(K[0, 0]),
           width=np.int64(cfg.width), height=np.int64(cfg.height),
           mesh_path=str(mesh_path), lighting=str(m.lighting),
           normalize_object=bool(m.normalize_object),
           cycles_samples=np.int64(m.cycles_samples),
           render_engine=str(m.render_engine), device=str(m.device))

  cmd = [str(m.blender_bin), "--background", "--python",
         os.path.join(HERE, "orbit_render_mesh.py"), "--", job, out]
  log.info("mesh render: %s", " ".join(cmd))
  r = subprocess.run(cmd)
  if r.returncode != 0 or not os.path.exists(out):
    raise SystemExit(f"Blender mesh render failed (exit {r.returncode})")
  return out


def _frame_metrics(mesh_f, gs_f, device):
  """mesh_f, gs_f: (H,W,3) float [0,1]. Returns (psnr, ssim, mae)."""
  diff = np.abs(mesh_f - gs_f)
  mae = float(diff.mean())
  mse = float((diff ** 2).mean())
  psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
  a = torch.from_numpy(np.ascontiguousarray(mesh_f)).permute(2, 0, 1)[None].to(device)
  b = torch.from_numpy(np.ascontiguousarray(gs_f)).permute(2, 0, 1)[None].to(device)
  s = float(ssim(a, b))
  return psnr, s, mae


def _diff_heatmap(mesh_f, gs_f, gain):
  """Per-pixel mean abs-diff -> magma RGB uint8, scaled by `gain`."""
  from matplotlib import colormaps
  d = np.clip(np.abs(mesh_f - gs_f).mean(-1) * float(gain), 0.0, 1.0)
  return (colormaps["magma"](d)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def compare_panels(cfg, gs_source, mesh_npy, device, metrics_out):
  """Generator: one composited comparison panel (uint8) per frame, streamed so
  the encoder never holds them all. `gs_source` is anything indexable to
  (H,W,3) uint8 (a list, or an mmap'd .npy). `metrics_out` is a list this
  fills with a per-frame dict as it goes; read it once the generator is done."""
  from PIL import Image, ImageDraw

  names = [str(p) for p in cfg.compare.panels]
  W, H = int(cfg.width), int(cfg.height)
  gain = float(cfg.compare.diff_gain)
  overlay = bool(cfg.compare.overlay_metrics)
  strip = 32 if overlay else 16          # a label row + (optionally) a metrics row
  bg = np.asarray(list(cfg.background), np.float32)

  mesh = np.load(mesh_npy, mmap_mode="r")               # (N,H,W,4) uint8 on disk
  n = len(gs_source)
  if len(mesh) != n:
    raise SystemExit(f"mesh render has {len(mesh)} frames, orbit has {n}")

  for i in range(n):
    mrgba = np.asarray(mesh[i], np.float32) / 255.0
    a = mrgba[..., 3:4]
    mf = mrgba[..., :3] * a + (1.0 - a) * bg            # composite over the same bg
    gs_u8 = np.asarray(gs_source[i], np.uint8)
    gf = gs_u8.astype(np.float32) / 255.0

    psnr, s, mae = _frame_metrics(mf, gf, device)
    metrics_out.append({"frame": i, "psnr": psnr, "ssim": s, "mae": mae})

    tiles = {"mesh": (np.clip(mf, 0, 1) * 255 + 0.5).astype(np.uint8),
             "gsplat": gs_u8,
             "diff": _diff_heatmap(mf, gf, gain)}
    row = np.concatenate([tiles[nm] for nm in names], axis=1)

    canvas = Image.new("RGB", (row.shape[1], H + strip), (16, 16, 18))
    canvas.paste(Image.fromarray(row), (0, strip))
    d = ImageDraw.Draw(canvas)
    for j, nm in enumerate(names):
      label = f"diff x{gain:g}" if nm == "diff" else ("3DGS" if nm == "gsplat" else nm)
      d.text((j * W + 6, 3), label, fill=(230, 230, 230))
    if overlay:
      txt = f"frame {i:>4}    PSNR {psnr:5.2f} dB    SSIM {s:.4f}    MAE {mae:.4f}"
      d.text((6, 17), txt, fill=(150, 210, 255))
    yield np.asarray(canvas)

    if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
      log.info("compared %d/%d frames", i + 1, n)


def log_metrics_summary(metrics):
  n = len(metrics)
  fin = [m["psnr"] for m in metrics if m["psnr"] != float("inf")]
  log.info("comparison over %d frames: mean PSNR %.2f dB  SSIM %.4f  MAE %.4f",
           n, (sum(fin) / len(fin)) if fin else float("inf"),
           sum(m["ssim"] for m in metrics) / n, sum(m["mae"] for m in metrics) / n)


def write_metrics_csv(path, metrics):
  with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["frame", "psnr", "ssim", "mae"])
    w.writeheader()
    for m in metrics:
      w.writerow({**m, "psnr": f"{m['psnr']:.6f}", "ssim": f"{m['ssim']:.6f}",
                  "mae": f"{m['mae']:.6f}"})
  return path


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def write_frames_dir(frames, out_dir):
  from PIL import Image
  os.makedirs(out_dir, exist_ok=True)
  for i, fr in enumerate(frames):
    Image.fromarray(fr).save(os.path.join(out_dir, f"frame_{i:06d}.png"))
  return out_dir


def write_mp4(frames, path, fps, crf):
  """H.264 .mp4 via imageio's ffmpeg backend. `imageio-ffmpeg` ships a static
  ffmpeg binary, so this needs nothing on the system PATH."""
  import imageio.v2 as imageio
  writer = imageio.get_writer(
    path, format="FFMPEG", mode="I", fps=float(fps),
    codec="libx264", macro_block_size=1,        # don't silently resize our frames
    pixelformat="yuv420p",                      # broad player compatibility
    ffmpeg_params=["-crf", str(int(crf)), "-preset", "medium"],
  )
  try:
    for fr in frames:
      writer.append_data(np.ascontiguousarray(fr))
  finally:
    writer.close()
  return path


def resolve_output(cfg, comparing):
  fmt = str(cfg.format).lower()
  if fmt not in ("mp4", "frames"):
    raise SystemExit(f"unknown format {fmt!r} (want mp4 or frames)")
  ext = {"mp4": ".mp4", "frames": ".frames"}[fmt]
  stem = ".orbit.compare" if comparing else ".orbit"
  out = cfg.output_path or (os.path.splitext(cfg.model_path)[0] + stem + ext)
  return fmt, out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="orbit")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  np.random.seed(int(cfg.seed))
  torch.manual_seed(int(cfg.seed))

  mesh_path = resolve_mesh_path(cfg)
  comparing = mesh_path is not None
  fmt, out = resolve_output(cfg, comparing=comparing)
  if fmt == "mp4" and (int(cfg.width) % 2 or int(cfg.height) % 2):
    cfg.width, cfg.height = int(cfg.width) + int(cfg.width) % 2, int(cfg.height) + int(cfg.height) % 2
    log.warning("mp4 (yuv420p) needs even dimensions; using %dx%d", int(cfg.width), int(cfg.height))

  with timed("load"):
    g = load_model(cfg)                              # numpy only -- no CUDA yet
  log.info("%d Gaussians from %s", len(g["means"]), cfg.model_path)

  poses, K, dist, center, n = orbit_poses(cfg, g["means"])
  log.info("%d frames  %dx%d  elevation %.1f deg  orbit %.1f deg  dist %.3f  look_at %s",
           n, int(cfg.width), int(cfg.height), float(cfg.elevation_deg),
           float(cfg.orbit_deg), dist, center.tolist())

  with tempfile.TemporaryDirectory() as workdir:
    # Mesh render FIRST, before torch touches CUDA -- Blender then has the GPU
    # (and RAM) to itself instead of fighting a live gsplat context.
    mesh_npy = None
    if comparing:
      log.info("mesh comparison against %s", mesh_path)
      with timed("render mesh"):
        mesh_npy = render_mesh_frames(cfg, poses, K, mesh_path, workdir)

    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device != cfg.device:
      log.warning("cuda not available, falling back to cpu")

    with timed("render gsplat"):
      gs = render_frames(cfg, g, poses, K, device,
                         out_npy=os.path.join(workdir, "gs.npy") if comparing else None)
    del g
    if device == "cuda":
      torch.cuda.empty_cache()

    frames, metrics = gs, None
    if comparing:
      metrics = []
      # generator: each panel is built + encoded one at a time, filling
      # `metrics` as it runs -- both inputs stay mmap'd on disk.
      frames = compare_panels(cfg, np.load(gs, mmap_mode="r"), mesh_npy, device, metrics)

    with timed("encode"):
      if fmt == "frames":
        write_frames_dir(frames, out)
      else:
        write_mp4(frames, out, cfg.fps, cfg.crf)
      if metrics is not None:
        log_metrics_summary(metrics)
        csv_path = write_metrics_csv(os.path.splitext(out)[0] + ".csv", metrics)
        log.info("wrote %s", csv_path)

  log.info("wrote %s", out)


if __name__ == "__main__":
  main()
