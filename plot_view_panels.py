#!/usr/bin/env python3
"""Per-view panel: an RGB strip (render + WT input) on top, then a grid with
one COLUMN per depth-peel layer and rows GT depth | WT depth | |delta| |
mask disagreement, for one or more views of a render_objaverse render vs its
wt_infer_layers.py prediction.

Both depth panels are RAW by default (`--align none`): the point of these
views is to see how well the geometry we generate already matches what World
Tracing predicts, in absolute terms. Scale/shift alignment (`--align
median|scale|affine|mad`, fit on layer 0 and applied to every layer) fits
that mismatch away, so it's opt-in only -- use it to inspect residual
*shape* error after the scale is taken out.

All the depth panels (GT + WT, every layer) share one colour range; all the
|delta| panels share another. The `mask disagreement` row flags where the
two validity masks disagree at that layer: red where WT predicts a surface
the render doesn't have, blue where the render has one WT missed, white
where they agree.

The title also carries the symmetric Chamfer distance (raw, no alignment)
between the render's unprojected depth-peel point cloud and WT's predicted
XYZ -- overall, and per layer in each row's label.

    python plot_view_panels.py bla/obj_rand40.h5 bla/obj_rand40.h5.wt.h5 --views 1 8 12 24
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from compare_wt_depth import _align, _layer0_depth_gt, _layer0_depth_pred
from util import intrinsics_name

MASK_EXTRA = "#d7263d"  # WT predicts a surface the render doesn't have
MASK_MISS = "#1f6feb"   # render has a surface WT missed


def _mask(a, valid):
  return np.where(valid, a, np.nan)


def _unproject(depth_hw, K):
  """depth_hw: (H, W), -1 = no hit. K: 3x3 pinhole. Returns (N, 3) points in
  OpenCV camera space (X right, Y down, Z fwd) -- the frame WT predicts in."""
  hit = depth_hw >= 0
  vv, uu = np.nonzero(hit)
  d = depth_hw[hit].astype(np.float64)
  k_inv = np.linalg.inv(np.asarray(K, np.float64))
  rays = np.column_stack([uu, vv, np.ones(uu.shape[0])]) @ k_inv.T
  return (rays * d[:, None]).astype(np.float32)


def _xyz_cloud(points_hwlc):
  """points_hwlc: (H, W, L, 3) or (H, W, 3), NaN = invalid. Returns (N, 3)."""
  flat = points_hwlc.reshape(-1, 3)
  return flat[~np.isnan(flat).any(axis=1)]


def _chamfer(a, b, cap=40000):
  """Symmetric Chamfer distance: mean_a min_b|a-b| + mean_b min_a|a-b|.
  Random-subsampled to `cap` points per side for speed; NaN if either side
  is empty."""
  if len(a) == 0 or len(b) == 0:
    return np.nan
  from scipy.spatial import cKDTree

  rng = np.random.default_rng(0)
  if len(a) > cap:
    a = a[rng.choice(len(a), cap, replace=False)]
  if len(b) > cap:
    b = b[rng.choice(len(b), cap, replace=False)]
  d_ab, _ = cKDTree(b).query(a)
  d_ba, _ = cKDTree(a).query(b)
  return float(d_ab.mean() + d_ba.mean())


def _disagreement_rgb(gt_present, wt_present):
  """White where the two validity masks agree, MASK_EXTRA where only WT has
  a value, MASK_MISS where only the render does."""
  from matplotlib.colors import to_rgb

  img = np.ones(gt_present.shape + (3,), np.float32)
  img[gt_present & ~wt_present] = to_rgb(MASK_MISS)
  img[~gt_present & wt_present] = to_rgb(MASK_EXTRA)
  return img


def panel(rgb_r, rgb_w, layers, title, out, wt_label="depth WT (raw)", crop=True):
  """layers: list of dicts {idx, gt, gtv, wt, wtv, both, absrel, n}, one per
  depth-peel layer to draw (gt/wt/gtv/wtv/both are H x W arrays).

  crop: trim every panel to the object's bounding box so the columns sit
  flush. Images are never stretched -- the figure just gets taller for a
  tall thin object. False keeps the full frame."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec
  from matplotlib.patches import Patch

  plt.rcParams.update({"font.size": 15})

  # the RGB render can be a higher resolution than the depth peel (split
  # width/height vs depth_width/depth_height); the crop window below is in
  # depth-peel pixels, so resample both RGB strips onto that grid first.
  dh, dw = layers[0]["gt"].shape

  def _fit(img):
    if img is None or img.shape[:2] == (dh, dw):
      return img
    yi = (np.arange(dh) * img.shape[0] / dh).astype(int)
    xi = (np.arange(dw) * img.shape[1] / dw).astype(int)
    return img[yi][:, xi]

  rgb_r, rgb_w = _fit(rgb_r), _fit(rgb_w)

  # shared ranges: depths over every GT+WT valid pixel of every layer,
  # deltas over every shared-valid pixel of every layer.
  depth_pool = np.concatenate(
    [L["gt"][L["gtv"] & (L["gt"] > 0)] for L in layers]
    + [L["wt"][L["wtv"]] for L in layers])
  delta_pool = np.concatenate([np.abs(L["gt"] - L["wt"])[L["both"]] for L in layers])
  vmin, vmax = np.percentile(depth_pool, [1, 99]) if depth_pool.size else (0.0, 1.0)
  dmax = float(np.percentile(delta_pool, 98)) if delta_pool.size else 1.0

  # crop window: the object's bounding box (union of GT/WT valid pixels over
  # all layers) + a small margin, or the full frame.
  h, w = layers[0]["gt"].shape
  if crop:
    occ = np.zeros((h, w), bool)
    for L in layers:
      occ |= L["gtv"] | L["wtv"]
    ys, xs = np.where(occ)
  else:
    ys = xs = np.array([], int)
  if ys.size:
    pad = max(4, int(0.03 * max(h, w)))
    r0, r1 = ys.min() - pad, ys.max() + 1 + pad
    c0, c1 = xs.min() - pad, xs.max() + 1 + pad
  else:
    r0, r1, c0, c1 = 0, h, 0, w

  # clamp the crop window's aspect to [0.7, 1.6] by widening the short axis
  # (shows a little more surrounding space -- the image itself is never
  # stretched) so a very tall/thin or wide/flat object x (nL+1) rows doesn't
  # produce an absurd figure.
  bh, bw = r1 - r0, c1 - c0
  if bh > 1.6 * bw:
    g = (bh / 1.6 - bw) / 2; c0 -= g; c1 += g
  elif bw > bh / 0.7:
    g = (bw * 0.7 - bh) / 2; r0 -= g; r1 += g
  r0, c0 = max(0, r0), max(0, c0)
  r1, c1 = min(h, r1), min(w, c1)

  # transposed layout: rows = depth GT | depth WT | |delta| | mask
  # disagreement; columns = the source RGB, then one per depth-peel layer.
  # "RGB render" sits in the GT row (both come from the render), "RGB WT
  # input" in the WT row. Equal-aspect (never stretched) images; cells sized
  # to the crop window's aspect so columns sit flush.
  nL = len(layers)
  ncol_img = 1 + nL                 # RGB column + one per layer
  panel_ar = min(1.6, max(0.7, (r1 - r0) / (c1 - c0)))
  cell_w = 2.5 if nL >= 3 else 3.2
  row_h = cell_w * panel_ar
  label_w = 1.05                    # inches, left row-labels
  cbar_cax_w, cbar_lbl_w = 0.26, 0.95
  margin_top, margin_bot = 1.35, 0.3
  fig_w = label_w + cell_w * ncol_img + cbar_cax_w + cbar_lbl_w
  fig_h = margin_top + row_h * 4 + margin_bot
  fig = plt.figure(figsize=(fig_w, fig_h))

  grid = fig.add_gridspec(4, ncol_img + 1,
                          left=label_w / fig_w, right=1 - cbar_lbl_w / fig_w,
                          top=1 - margin_top / fig_h, bottom=margin_bot / fig_h,
                          width_ratios=[1] * ncol_img + [cbar_cax_w / cell_w],
                          height_ratios=[1] * 4, hspace=0.13, wspace=0.02)

  def show(ax, img, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)  # aspect "equal"
    ax.set_xlim(c0 - 0.5, c1 - 0.5)
    ax.set_ylim(r1 - 0.5, r0 - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
      s.set_visible(True); s.set_color("black"); s.set_linewidth(1.4)
    return im

  def label_only(ax, text):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
      s.set_visible(False)
    ax.set_ylabel(text, fontsize=13)

  gt_of = lambda L: _mask(L["gt"], L["gtv"] & (L["gt"] > 0))
  rows = [
    ("depth GT",      rgb_r, "RGB render",
     lambda L: (gt_of(L), "turbo", vmin, vmax)),
    (wt_label,        rgb_w, "RGB WT input",
     lambda L: (_mask(L["wt"], L["wtv"]), "turbo", vmin, vmax)),
    ("|delta|",       None, None,
     lambda L: (_mask(np.abs(L["gt"] - L["wt"]), L["both"]), "turbo", 0.0, dmax)),
    ("mask disagree", None, None,
     lambda L: (_disagreement_rgb(L["gtv"] & (L["gt"] > 0), L["wtv"]), None, None, None)),
  ]
  ims = {}
  for ri, (rlabel, rgb, rgb_title, fn) in enumerate(rows):
    ax0 = fig.add_subplot(grid[ri, 0])
    if rgb is not None:
      show(ax0, rgb)
      ax0.set_title(rgb_title, fontsize=11, pad=5)
      ax0.set_ylabel(rlabel, fontsize=13)
    else:
      label_only(ax0, rlabel)
    for ci, L in enumerate(layers, start=1):
      ax = fig.add_subplot(grid[ri, ci])
      ims[ri] = show(ax, *fn(L))
      if ri == 0:
        ar = f"AbsRel {L['absrel']:.3f}" if np.isfinite(L["absrel"]) else "AbsRel --"
        cd = f"CD {L['cd']:.4f}" if np.isfinite(L.get("cd", np.nan)) else "CD --"
        ax.set_title(f"layer {L['idx']}\n{ar}  ·  {cd}\nn={L['n']}", fontsize=10, pad=5)

  fig.colorbar(ims[0], cax=fig.add_subplot(grid[0:2, ncol_img]), label="depth")
  fig.colorbar(ims[2], cax=fig.add_subplot(grid[2, ncol_img]), label="|delta|")
  axl = fig.add_subplot(grid[3, ncol_img]); axl.axis("off")
  axl.legend(handles=[Patch(facecolor=MASK_EXTRA, edgecolor="0.4", label="WT extra"),
                      Patch(facecolor=MASK_MISS, edgecolor="0.4", label="WT missing")],
             loc="center left", fontsize=9, frameon=False, handlelength=1.1,
             borderaxespad=0)
  fig.suptitle(title, y=1 - 0.42 / fig_h, fontsize=min(15, fig_w * 1.6))
  fig.savefig(out, dpi=100)
  plt.close(fig)
  print(f"[fig] {out}")


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5")
  p.add_argument("wt_h5")
  p.add_argument("--views", type=int, nargs="+", required=True)
  p.add_argument("--layers", type=int, nargs="+", default=None,
                 help="Which depth-peel layers to show (default: all of them, "
                      "empty ones included)")
  p.add_argument("--align", choices=["none", "median", "scale", "affine", "mad"], default="none",
                 help="Scale/shift-fit WT onto GT (on layer 0, applied to all "
                      "layers) before display. Default none -- these views are "
                      "for seeing the raw match to WT.")
  p.add_argument("--crop", action="store_true", default=True,
                 help="Crop panels to the object bounding box so columns sit "
                      "flush (default). --no-crop keeps the full frame.")
  p.add_argument("--no-crop", dest="crop", action="store_false")
  p.add_argument("--chamfer", action="store_true", default=True,
                 help="Compute the raw Chamfer distance between the render's "
                      "unprojected depth peel and WT's XYZ (default). "
                      "--no-chamfer skips the KD-tree work.")
  p.add_argument("--no-chamfer", dest="chamfer", action="store_false")
  p.add_argument("--out-prefix", default=None,
                 help="default: <wt_h5>.panel  ->  <prefix>.viewN.png")
  args = p.parse_args()
  prefix = args.out_prefix or f"{args.wt_h5}.panel"

  with h5py.File(args.render_h5, "r") as rf, h5py.File(args.wt_h5, "r") as wf:
    mesh_index = rf["mesh_index"][:] if "mesh_index" in rf else None
    K_name = intrinsics_name(rf, "depth", None) if args.chamfer else None
    K_ds = rf[K_name] if K_name else None
    for v in args.views:
      dp = rf["depth_peel"][v]      # (H, W, L)
      pts = wf["points"][v]         # (H, W, L, 3)
      n_layers = min(dp.shape[2], pts.shape[2])

      # one alignment for the view, fit on layer 0's shared valid pixels
      gt0, gtv0 = _layer0_depth_gt(dp, 0)
      pr0, prv0 = _layer0_depth_pred(pts, 0)
      both0 = gtv0 & prv0 & np.isfinite(pr0) & (gt0 > 0)
      s, t = _align(pr0[both0], gt0[both0], args.align)

      # Chamfer distance is always on the RAW clouds (a Z-only scale/shift
      # isn't a valid point-cloud transform), regardless of --align.
      K = K_ds[v] if K_ds is not None else None
      wt_cloud_all = _xyz_cloud(pts) if K is not None else None
      cd_all = (_chamfer(np.concatenate([_unproject(dp[..., li], K)
                                         for li in range(n_layers)], axis=0),
                         wt_cloud_all)
                if K is not None else np.nan)

      layers = []
      for li in (args.layers if args.layers is not None else range(n_layers)):
        gt, gtv = _layer0_depth_gt(dp, li)
        pr, prv = _layer0_depth_pred(pts, li)
        wt = s * pr + t
        wtv = prv & np.isfinite(wt)
        both = gtv & wtv & (gt > 0)
        absrel = (float(np.mean(np.abs(wt[both] - gt[both]) / gt[both]))
                  if both.any() else np.nan)
        cd = (_chamfer(_unproject(dp[..., li], K), _xyz_cloud(pts[:, :, li, :]))
              if K is not None else np.nan)
        layers.append(dict(idx=li, gt=gt, gtv=gtv, wt=wt, wtv=wtv, both=both,
                           absrel=absrel, n=int(both.sum()), cd=cd))

      m = f"mesh {int(mesh_index[v])}" if mesh_index is not None else ""
      tags = [f"view {v}", m, f"{len(layers)} layer(s)"]
      if args.align != "none":
        tags.append(f"align={args.align} (s={s:.3g}, t={t:.3g})")
      if np.isfinite(cd_all):
        tags.append(f"Chamfer(all) {cd_all:.4f}")
      title = "    ".join(t for t in tags if t)
      wt_label = "depth WT (raw)" if args.align == "none" else f"depth WT ({args.align}-aligned)"
      panel(rf["images"][v], wf["images"][v], layers, title,
            f"{prefix}.view{v}.png", wt_label=wt_label, crop=args.crop)


if __name__ == "__main__":
  main()
