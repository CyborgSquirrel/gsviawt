#!/usr/bin/env python3
"""Sanity-check the Flash3D inference setup.

Exercises the same code path as flash3d_infer.py end to end, but needs no
input image or output directory: it uses a bundled demo image (an indoor
room, from the official HF Space's own examples -- Flash3D is a
RealEstate10K scene model, so an object photo like splatter-image's demo
car would be out of distribution here) and only checks that each stage
produces something plausible. Run this after building/rebuilding the image,
or after touching anything under flash3d/, to confirm the vendored CUDA
rasterizer, the self-contained UniDepth v1 copy, the HuggingFace checkpoint
download, and the model forward+render pass all still work.

Exits non-zero on the first failed stage.

    python check_flash3d_setup.py
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_DEMO_IMAGE = _ROOT / "flash3d" / "demo_examples" / "bedroom_01.png"


def _step(name):
    print(f"[ ] {name}...", end=" ", flush=True)

    def done(ok=True, detail=""):
        print(("OK" if ok else "FAIL") + (f" ({detail})" if detail else ""))
        if not ok:
            sys.exit(1)

    return done


def main():
    done = _step("import diff_gaussian_rasterization (vendored w-pose CUDA rasterizer)")
    try:
        import diff_gaussian_rasterization  # noqa: F401
    except ImportError as e:
        done(False, str(e))
    done()

    done = _step("import flash3d_infer (pulls in the vendored unidepth/ + networks/ packages)")
    import flash3d_infer as fi

    done()

    done = _step("NystromBlock falls back to exact attention without xformers")
    from unidepth.layers.nystrom_attention import NystromAttention

    if NystromAttention is not None:
        done(detail="xformers.components is actually importable here -- using the real Nystrom approximation")
    else:
        done(detail="xformers.components unavailable, as expected -- falling back to exact SDPA (see nystrom_attention.py)")

    done = _step("CUDA available")
    if not torch.cuda.is_available():
        done(False, "torch.cuda.is_available() is False")
    device = torch.device("cuda:0")
    done()

    done = _step("download + load flash3d re10k_v1 checkpoint from HuggingFace")
    model, model_cfg = fi._load_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params < 1_000_000:
        done(False, f"suspiciously small model ({n_params} params)")
    done(detail=f"{n_params:,} params")

    done = _step(f"preprocess demo image ({_DEMO_IMAGE.name})")
    image = fi._preprocess(_DEMO_IMAGE, model_cfg).to(device)
    pad = model_cfg.dataset.pad_border_aug
    expected_hw = (model_cfg.dataset.height + 2 * pad, model_cfg.dataset.width + 2 * pad)
    if tuple(image.shape) != (1, 3, *expected_hw):
        done(False, f"unexpected shape {tuple(image.shape)}, wanted (1, 3, {expected_hw[0]}, {expected_hw[1]})")
    done()

    done = _step("forward pass (image -> layered Gaussians via UniDepth + ResNet head)")
    with torch.no_grad():
        outputs = model({("color_aug", 0, 0): image})
    reconstruction, (h, w) = fi._extract_gaussians(outputs, model_cfg, crop=True)
    n_gaussians = reconstruction["xyz"].shape[0]
    expected_n = model_cfg.model.gaussians_per_pixel * h * w
    if n_gaussians != expected_n:
        done(False, f"expected {expected_n} Gaussians ({model_cfg.model.gaussians_per_pixel} layers x "
                    f"{h}x{w}), got {n_gaussians}")
    done(detail=f"{n_gaussians} Gaussians across {model_cfg.model.gaussians_per_pixel} layers")

    done = _step("export .ply (Z-up remap + opacity/scale un-activation for orbit_video.py)")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ply_path = Path(tmp) / "check.ply"
        fi._export_ply(reconstruction, str(ply_path), min_opacity=0.5)
        if not ply_path.exists() or ply_path.stat().st_size == 0:
            done(False, "no .ply written")
    done()

    done = _step("render the reconstruction (preview wiggle loop, one frame)")
    reconstruction_render, image_hw = fi._extract_gaussians(outputs, model_cfg, crop=False)
    K = outputs[("K_src", 0, 0)][0]
    with torch.no_grad():
        frames = fi._render_loop(reconstruction_render, model_cfg, K, image_hw,
                                  fi.OmegaConf.create({"num_loop_views": 1, "max_yaw_deg": 10.0,
                                                        "znear": 0.01, "zfar": 100.0}), device)
    render = frames[0]
    if render.std() < 1e-4:
        done(False, f"render looks blank (std={render.std():.2e})")
    done(detail=f"{render.shape}, std={render.std():.3f}")

    print("\nSetup OK -- flash3d inference is working end to end.")


if __name__ == "__main__":
    main()
