#!/usr/bin/env python3
"""Compare two render_objaverse.py h5 files row by row.

Meant for A/B-ing render settings (Cycles vs EEVEE, ...) on identical scenes:
both files must hold the same rows (same mesh_index, same camera_pose), e.g.
the second made from the first with `render_objaverse.py source_path=<first>
source_rerender_rgb=true render_engine=...`. This is asserted, so a silently
different scene (say, a mesh glob that has drifted) can't produce a bogus
comparison.

Per row it reports, on RGB composited over black (the film is transparent
and the background black, so this is what the pixels look like):

  psnr       PSNR in dB (255 peak) over all pixels
  mae        mean absolute difference, 0-255
  alpha_iou  IoU of the alpha > 127 masks
  fg_mae     MAE restricted to pixels in either mask (the object)
  depth_max  max |a - b| of depth_peel (NaN-aware); ~0 (float noise at a few
             edge pixels) when only the RGB engine differs, since the depth
             peel always runs on Cycles

and writes metrics.csv, summary.json and raw side-by-side strips of the
worst and of evenly spaced rows. Config is Hydra (conf/compare_renders.yaml);
run in the container venv:

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python \\
        compare_renders.py a_path=/app/bla/x.h5 b_path=/app/bla/x_eevee.h5 \\
        out_dir=/app/bla/x_eevee_cmp
"""

import csv
import json
import os

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageDraw


def over_black(rgba):
  return rgba[..., :3].astype(np.float32) * (rgba[..., 3:4].astype(np.float32) / 255.0)


def row_metrics(ra, rb, da, db):
  ca, cb = over_black(ra), over_black(rb)
  err = (ca - cb) ** 2
  mse = float(err.mean())
  ma, mb = ra[..., 3] > 127, rb[..., 3] > 127
  union = ma | mb
  diff = np.abs(ca - cb).mean(-1)
  both_nan = np.isnan(da) & np.isnan(db)
  d = np.where(both_nan, 0.0, np.abs(da - db))
  return dict(
    psnr=float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse),
    mae=float(np.abs(ca - cb).mean()),
    alpha_iou=float((ma & mb).sum() / union.sum()) if union.any() else 1.0,
    fg_mae=float(diff[union].mean()) if union.any() else 0.0,
    depth_max=float(np.nanmax(d)) if d.size else 0.0,
  )


def strip(ra, rb, cfg, label):
  ca, cb = over_black(ra), over_black(rb)
  gain = float(cfg.diff_gain)
  panels = [ca, cb, np.clip(np.abs(ca - cb) * gain, 0, 255)]
  names = [cfg.a_name, cfg.b_name, f"|diff| x{gain:g}"]
  t = int(cfg.thumb)
  tiles = []
  for p, n in zip(panels, names):
    im = Image.fromarray(p.astype(np.uint8)).resize((t, t), Image.LANCZOS)
    ImageDraw.Draw(im).text((6, 4), n, fill=(255, 255, 0))
    tiles.append(np.asarray(im))
  out = Image.fromarray(np.concatenate(tiles, 1))
  ImageDraw.Draw(out).text((6, t - 14), label, fill=(255, 255, 0))
  return np.asarray(out)


@hydra.main(version_base=None, config_path="conf", config_name="compare_renders")
def main(cfg: DictConfig):
  os.makedirs(cfg.out_dir, exist_ok=True)
  with h5py.File(cfg.a_path, "r") as a, h5py.File(cfg.b_path, "r") as b:
    n = b["images"].shape[0] if cfg.max_rows is None else int(cfg.max_rows)
    if a["images"].shape[0] < n:
      raise ValueError(f"a has {a['images'].shape[0]} rows, need {n}")
    if not np.array_equal(a["mesh_index"][:n], b["mesh_index"][:n]):
      raise ValueError("mesh_index differs between the two files")
    # not array_equal: a source_path re-render rebuilds the camera from the
    # stored matrix, which round-trips only to float32 rounding (~1e-7)
    if not np.allclose(a["camera_pose"][:n], b["camera_pose"][:n], atol=1e-5):
      raise ValueError("camera_pose differs between the two files")
    ma = [p for p in a["mesh_paths"].asstr()[()]]
    mb = [p for p in b["mesh_paths"].asstr()[()]]
    k = int(b["mesh_index"][:n].max()) + 1
    if ma[:k] != mb[:k]:
      raise ValueError("mesh_paths differ between the two files")
    for f, name in ((a, "a"), (b, "b")):
      cj = json.loads(f.attrs["config_json"])
      print(f"{name}: render_engine={cj.get('render_engine')} "
            f"eevee={cj.get('eevee')} cycles_samples={cj.get('cycles_samples')}")

    rows = []
    for i in range(n):
      m = row_metrics(a["images"][i], b["images"][i], a["depth_peel"][i], b["depth_peel"][i])
      rows.append(dict(row=i, mesh=os.path.basename(ma[int(a["mesh_index"][i])]), **m))

  with open(os.path.join(cfg.out_dir, "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

  def stat(key):
    v = np.array([r[key] for r in rows], np.float64)
    v = v[np.isfinite(v)] if key == "psnr" else v
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                min=float(v.min()), max=float(v.max()))
  summary = {k: stat(k) for k in ("psnr", "mae", "alpha_iou", "fg_mae", "depth_max")}
  summary["rows"] = n
  with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))

  order = sorted(range(n), key=lambda i: rows[i]["psnr"])
  spaced = np.unique(np.linspace(0, n - 1, int(cfg.num_spaced)).astype(int)) if n else []
  with h5py.File(cfg.a_path, "r") as a, h5py.File(cfg.b_path, "r") as b:
    for tag, idxs in (("worst", order[:int(cfg.num_worst)]), ("spaced", list(spaced))):
      strips = [strip(a["images"][i], b["images"][i], cfg,
                      f"row {i} {rows[i]['mesh'][:8]} psnr {rows[i]['psnr']:.1f}dB")
                for i in idxs]
      if strips:
        Image.fromarray(np.concatenate(strips, 0)).save(os.path.join(cfg.out_dir, f"{tag}.png"))


if __name__ == "__main__":
  main()
