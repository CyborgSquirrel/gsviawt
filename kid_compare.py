"""KID of eval_diffusion.py's kid features (<out>/[cs_<name>/][steps<N>/]kid_features.npz; conf/kid_compare.yaml), per set, x 100,
with k(x, y) = (x.y / d + 1)^3 (d = 1024 for DINOv2; the full estimators, no subsets), gen = the samples, gt = the GT renders:
  kid_plain  the standard unbiased MMD^2 (Gretton et al. 2012 eq. 3): XX and YY over all pairs of distinct images, XY over all pairs
  kid_objex  object-excluded: every pair of images of the SAME mesh dropped in XX, YY and XY (the matched gen / GT of a mesh
             included) -- eq. 4 with the meshes as the units, so it stays unbiased for the view-marginal MMD^2 under the dependence
             within a mesh (its views, the gen conditioned on its GT); with V views per mesh in both sets,
             sum_{o != o'} (A + B - C - C^T)_{oo'} / (V^2 N (N - 1)) of the per-mesh block sums A, B, C of Kxx, Kyy, Kxy
Both with a leave-one-mesh-out jackknife SE; with b (a second feature file over the SAME (set, mesh, view) images, e.g. another
run or the same run with other noise): the paired difference a - b and its jackknife SE (the GT term cancels). Writes <out>.json
and <out>.csv (out null: <a's dir>/kid[_vs_<b's last two dir names>]). The unit of a set is its mesh: N_meshes, N_images per set."""
import csv
import json
import os

import hydra
import numpy as np

ESTIMATORS = ("kid_plain", "kid_objex")
SCALE = 1e2   # x 100, the TRELLIS convention for KD with DINOv2 features (arXiv:2412.01506)


def kernel(x, y):
  """The cubic polynomial kernel (x.y / d + 1)^3, float64."""
  x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
  return (x @ y.T / x.shape[1] + 1) ** 3


def blocks(gen, gt, units):
  """Per-unit block sums of one set: A = Kxx, B = Kyy, C = Kxy summed over the images of units (o, o') (N x N), the traces of Kxx /
  Kyy per unit, and the images per unit (gen and gt have the same rows: units[i] of both)."""
  u, inv = np.unique(units, return_inverse=True)
  onehot = np.zeros((len(units), len(u)))
  onehot[np.arange(len(units)), inv] = 1
  kxx, kyy, kxy = kernel(gen, gen), kernel(gt, gt), kernel(gen, gt)
  return {"A": onehot.T @ kxx @ onehot, "B": onehot.T @ kyy @ onehot, "C": onehot.T @ kxy @ onehot,
          "trA": onehot.T @ np.diag(kxx), "trB": onehot.T @ np.diag(kyy), "n": onehot.sum(0)}


def _quad(M, w):
  """w^T M w for each row of w (a batch of unit weights)."""
  return np.einsum("ko,op,kp->k", w, M, w)


def estimates(b, w):
  """{estimator: (K,) values} on the units kept by each row of w (K, N) of 0 / 1 weights."""
  n = w @ b["n"]
  aa, bb, cc = _quad(b["A"], w), _quad(b["B"], w), _quad(b["C"], w)
  plain = (aa - w @ b["trA"]) / (n * (n - 1)) + (bb - w @ b["trB"]) / (n * (n - 1)) - 2 * cc / (n * n)
  pairs = n * n - w @ (b["n"] ** 2)   # image pairs of distinct units
  objex = (aa - w @ np.diag(b["A"])) / pairs + (bb - w @ np.diag(b["B"])) / pairs - 2 * (cc - w @ np.diag(b["C"])) / pairs
  return {"kid_plain": plain, "kid_objex": objex}


def jackknife(full, loo):
  """(estimate, SE) of the leave-one-unit-out values loo (N,): SE = sqrt((N - 1) / N sum (loo - mean)^2)."""
  n = len(loo)
  return float(full), float(np.sqrt((n - 1) / n * np.square(loo - loo.mean()).sum()))


def set_estimates(b):
  """{estimator: (full value, (N,) leave-one-unit-out values)}."""
  n = len(b["n"])
  full, loo = estimates(b, np.ones((1, n))), estimates(b, 1 - np.eye(n))
  return {k: (full[k][0], loo[k]) for k in ESTIMATORS}


def load(path):
  """The feature file as a dict of arrays; one (checkpoint, weights) per file."""
  z = dict(np.load(path, allow_pickle=False))
  pairs = sorted(set(zip(z["checkpoint"].tolist(), z["weights"].tolist())))
  if len(pairs) != 1:
    raise SystemExit(f"{path}: features of {len(pairs)} (checkpoint, weights): give one per file")
  return z


def order(z, idx):
  """idx sorted by (mesh, view): the canonical image order of a set."""
  return idx[np.lexsort((z["view"][idx], z["mesh"][idx]))]


def compare(a, b=None, sets=None):
  """[{set, n_meshes, n_images, <estimator>, <estimator>_se, (with b) <estimator>_b / _se, delta_<estimator> / _se}] x 100."""
  out = []
  for s in (sets or sorted(set(a["set"].tolist()))):
    ia = order(a, np.flatnonzero(a["set"] == s))
    if not len(ia):
      raise SystemExit(f"no images of set {s!r}")
    ba = blocks(a["gen"][ia], a["gt"][ia], a["mesh"][ia])
    ea = set_estimates(ba)
    row = {"set": s, "n_meshes": len(ba["n"]), "n_images": len(ia)}
    for k, (f, loo) in ea.items():
      row[k], row[f"{k}_se"] = (SCALE * v for v in jackknife(f, loo))
    if b is not None:
      ib = order(b, np.flatnonzero(b["set"] == s))
      if not (len(ib) == len(ia) and (b["mesh"][ib] == a["mesh"][ia]).all() and (b["view"][ib] == a["view"][ia]).all()):
        raise SystemExit(f"set {s}: the two feature files do not have the same (mesh, view) images")
      gt_diff = float(np.abs(b["gt"][ib] - a["gt"][ia]).max())
      eb = set_estimates(blocks(b["gen"][ib], b["gt"][ib], b["mesh"][ib]))
      row["gt_max_abs_diff"] = gt_diff   # ~0: the same GT features (else the GT term does not cancel exactly)
      for k, (f, loo) in eb.items():
        row[f"{k}_b"], row[f"{k}_b_se"] = (SCALE * v for v in jackknife(f, loo))
        row[f"delta_{k}"], row[f"delta_{k}_se"] = (SCALE * v for v in jackknife(ea[k][0] - f, ea[k][1] - loo))
    out.append(row)
  return out


@hydra.main(version_base=None, config_path="conf", config_name="kid_compare")
def main(cfg):
  a = load(cfg.a)
  b = load(cfg.b) if cfg.b is not None else None
  rows = compare(a, b, list(cfg.sets) if cfg.sets is not None else None)
  vs = "_".join(os.path.dirname(os.path.abspath(cfg.b)).split(os.sep)[-2:]) if b is not None else None   # e.g. <eval>_steps20
  out = cfg.out or os.path.join(os.path.dirname(os.path.abspath(cfg.a)), "kid" + (f"_vs_{vs}" if vs else ""))
  with open(f"{out}.json", "w") as f:
    json.dump({"a": cfg.a, "b": cfg.b, "scale": SCALE, "meta_a": json.loads(str(a["meta"])),
               "meta_b": json.loads(str(b["meta"])) if b is not None else None, "results": rows}, f, indent=1)
  with open(f"{out}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  for r in rows:
    line = f"{r['set']:>18} N_meshes {r['n_meshes']:3d} N_images {r['n_images']:4d}  " + "  ".join(
      f"{k} {r[k]:.3f} +- {r[k + '_se']:.3f}" for k in ESTIMATORS)
    if b is not None:
      line += "  |  b: " + "  ".join(f"{k} {r[k + '_b']:.3f} +- {r[k + '_b_se']:.3f}" for k in ESTIMATORS)
      line += "  |  a - b: " + "  ".join(f"{k} {r['delta_' + k]:+.3f} +- {r['delta_' + k + '_se']:.3f}" for k in ESTIMATORS)
    print(line)
  print(f"(x 100) wrote {out}.json, {out}.csv")


if __name__ == "__main__":
  main()
