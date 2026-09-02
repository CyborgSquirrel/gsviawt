#!/usr/bin/env python3
"""Run the World Tracing object model (r75b) on one view of a
capture_turntable.py render and dump its per-layer output as an HDF5
file laid out like renders.h5, for direct comparison against
debug_pointcloud.py's depth-peel output.

Mirrors world-tracing/examples/infer_rgba.py (model load -> preprocess ->
inference_diffusion) but skips the Rerun/.rrd visualisation entirely and
instead:
  * reads the input RGB + mask straight from our own renders.h5 (the same
    pixels/alpha capture_turntable.py wrote), instead of a saved PNG;
  * writes an HDF5 file with an 'images' dataset (the image actually fed
    to the model, i.e. after preprocess_rgba_for_model's crop/resize/bg
    blend) and a 'points' dataset (the predicted per-layer XYZ, one
    (H, W, L, 3) volume per image, NaN where invalid -- the vector
    analogue of renders.h5's depth_peel, which uses -1.0 as its scalar
    invalid marker).

Coordinate-frame note
----------------------
World Tracing predicts XYZ in its own camera space only (X right, Y down,
Z forward -- OpenCV/RDF convention); it has no notion of world pose, just
an intrinsics matrix recoverable post-hoc via
`wt.solve_intrinsics_from_xyz` (written to the 'intrinsics' dataset
below). That is the same frame as the `points_cv`
intermediate inside debug_pointcloud.py's `unproject_depth_peel`, i.e.
*before* the `pose` (camera-to-world) matrix gets applied -- so compare
against that, not the world-space output debug_pointcloud.py writes by
default.

Also note `preprocess_rgba_for_model` re-centers/rescales the object to
fill ~2/3 of the model's square input canvas (matching its Objaverse
training data), regardless of how tightly the Blender camera already
framed it (frame_object_robust uses a 5% margin, i.e. ~full-frame). So
don't expect pixel-for-pixel alignment with the raw render -- compare
overall shape/scale/extent, or solve for a similarity transform if you
need per-pixel error.

Usage
-----

    python wt_infer_layers.py bla/renders.h5 0 \
        --ckpt r75b --config r75b --out /tmp/wt_view0.h5
"""

from argparse import ArgumentParser

import h5py
import numpy as np
import torch


def rgba_from_render(hf, index):
  """Build an H,W,4 uint8 RGBA array for one view: alpha comes from the
  layer-0 depth-peel hit mask (same foreground definition
  debug_pointcloud.py uses for `layer_idx == 0`)."""
  image = hf["images"][index]  # (H, W, 3) uint8
  depth_peel = hf["depth_peel"][index]  # (H, W, L) float32, -1.0 = no hit
  alpha = np.where(depth_peel[..., 0] >= 0, 255, 0).astype(np.uint8)
  return np.concatenate([image, alpha[..., None]], axis=-1)


def main():
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("hdf5_path", help="Path to a capture_turntable.py renders.h5")
  parser.add_argument("index", type=int, help="View index within the HDF5")
  parser.add_argument("--ckpt", default="r75b", help="Checkpoint (config name / hf:// / local path)")
  parser.add_argument("--config", default="r75b", help="Model config (default: r75b, the object model)")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-steps", type=int, default=None, help="Override the config's default sampler steps")
  parser.add_argument("--out", default=None, help="Output .h5 path (default: <hdf5>.view<index>.wt.h5)")
  # Preprocessing knobs mirrored 1:1 from examples/infer_rgba.py's CLI
  # (wt.cli.add_common_args) so this matches "their procedure" exactly
  # instead of silently drifting from preprocess_rgba_for_model's own
  # (different!) defaults.
  parser.add_argument("--alpha-erode", type=int, default=0)
  parser.add_argument("--no-center-crop", action="store_true")
  parser.add_argument("--bg-color", type=str, default="128,128,128")
  parser.add_argument(
    "--bf16-weights-hack", action="store_true", default=False,
    help=(
      "HACK, off by default: cast the model's stored weights to bf16 "
      "after loading (in addition "
      "to the existing bf16 autocast around the forward pass). "
      "build_model_and_load_ckpt() only does model.to(device); autocast "
      "casts activations per-op but never touches how params are stored, "
      "so the 1.7B r75b params otherwise stay resident in fp32 (~6.8GB) "
      "for the whole run. That alone is most of a 10GB card's budget "
      "before any activations exist. This flag drops resident weight "
      "memory to ~3.4GB (verified: 7.48GB peak reserved at full 504x504, "
      "default 4-seed sweep, on a 10GB RTX 3080 -- flat across seeds, "
      "~2GB headroom). Off by default because it makes layernorm/softmax "
      "etc. run on bf16-stored params instead of the fp32-stored params "
      "autocast normally protects them with -- expected small numerical "
      "delta for a bf16-compute-validated model, not a proven zero-diff."
    ),
  )
  args = parser.parse_args()

  from wt import inference_diffusion, solve_intrinsics_from_xyz
  from wt.checkpoint import build_model_and_load_ckpt
  from wt.cli import parse_bg_color
  from wt.data import preprocess_rgba_for_model
  from wt.inference import _bypass_activation_checkpointing

  out_path = args.out or f"{args.hdf5_path}.view{args.index}.wt.h5"
  bg_color = parse_bg_color(args.bg_color)

  with h5py.File(args.hdf5_path, "r") as hf:
    rgba = rgba_from_render(hf, args.index)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[wt] config={args.config}, device={device}")
  model, cfg = build_model_and_load_ckpt(args.config, args.ckpt, device)
  if args.bf16_weights_hack:
    print("[wt] --bf16-weights-hack: casting stored weights to bf16 (see --help for the tradeoff)")
    model = model.to(torch.bfloat16)

  inference_kwargs = dict(cfg["inference_kwargs"])
  if args.num_steps is not None:
    inference_kwargs["num_steps"] = args.num_steps

  rgb_t, mask_t, intr_t = preprocess_rgba_for_model(
    rgba,
    image_size=cfg["image_size"],
    num_layers=cfg["model_kwargs"]["num_layers"],
    alpha_erode_px=args.alpha_erode,
    center_crop=not args.no_center_crop,
    bg_color=bg_color,
  )
  rgb_t, mask_t, intr_t = rgb_t.to(device), mask_t.to(device), intr_t.to(device)

  torch.manual_seed(args.seed)
  if device.type == "cuda":
    torch.cuda.manual_seed(args.seed)

  autocast_ctx = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda"
    else torch.autocast(device_type="cpu", enabled=False)
  )
  print(f"[wt] running diffusion (seed={args.seed}) ...")
  with torch.no_grad(), autocast_ctx, _bypass_activation_checkpointing(model):
    xyz_pred, mask_pred, _ = inference_diffusion(
      model, rgb_t, gt_mask=mask_t, use_gt_mask=True, intrinsics=intr_t,
      invalid_fill_mode="noise", **inference_kwargs,
    )
  xyz = xyz_pred[0].float().cpu().numpy()  # [L, H, W, 3]
  mask = mask_pred[0].cpu().numpy().astype(bool)  # [L, H, W]
  rgb_uint8 = (rgb_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

  K, fov_x = solve_intrinsics_from_xyz(xyz[0], mask[0], image_size=cfg["image_size"])
  print(f"[wt] solved K from layer-0 XYZ; fov_x ~= {fov_x:.1f} deg")

  # [L, H, W, 3] -> [H, W, L, 3], matching depth_peel's (H, W, L) axis
  # order; invalid entries get NaN instead of depth_peel's -1.0 sentinel
  # since a sentinel *vector* would collide with real geometry.
  points = np.transpose(xyz, (1, 2, 0, 3)).copy()
  points[~np.transpose(mask, (1, 2, 0))] = np.nan

  K = K if K is not None else intr_t[0].cpu().numpy()

  with h5py.File(out_path, "w") as out:
    images_ds = out.create_dataset("images", data=rgb_uint8[None])
    points_ds = out.create_dataset("points", data=points[None])
    out.create_dataset("intrinsics", data=K[None])
    out.create_dataset("seed", data=np.array([args.seed], dtype=np.int64))
    out.create_dataset("config", data=np.array([args.config], dtype=h5py.string_dtype()))
    images_shape, points_shape = images_ds.shape, points_ds.shape

  n_valid = int(mask.sum())
  print(f"[wt] wrote images{images_shape} points{points_shape} ({n_valid} valid) to {out_path}")


if __name__ == "__main__":
  main()
