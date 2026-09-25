#!/usr/bin/env python3
"""Sanity-check the Splatter Image inference setup.

Exercises the same code path as splatter_image_infer.py end to end, but
needs no input image or output directory: it uses a bundled demo image and
only checks that each stage produces something plausible. Run this after
building/rebuilding the image, or after touching anything under
splatter-image/, to confirm the vendored CUDA rasterizer, the HuggingFace
checkpoint download, and the model forward+render pass all still work.

Exits non-zero on the first failed stage.

    python check_splatter_image_setup.py
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_DEMO_IMAGE = _ROOT / "splatter-image" / "demo_examples" / "01_bigmac.png"


def _step(name):
    print(f"[ ] {name}...", end=" ", flush=True)

    def done(ok=True, detail=""):
        print(("OK" if ok else "FAIL") + (f" ({detail})" if detail else ""))
        if not ok:
            sys.exit(1)

    return done


def main():
    done = _step("import diff_gaussian_rasterization (vendored CUDA rasterizer)")
    try:
        import diff_gaussian_rasterization  # noqa: F401
    except ImportError as e:
        done(False, str(e))
    done()

    done = _step("import splatter_image_infer")
    import splatter_image_infer as si

    done()

    done = _step("CUDA available")
    if not torch.cuda.is_available():
        done(False, "torch.cuda.is_available() is False")
    device = torch.device("cuda:0")
    done()

    done = _step("download + load cars checkpoint from HuggingFace")
    model, model_cfg = si._load_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params < 1_000_000:
        done(False, f"suspiciously small model ({n_params} params)")
    done(detail=f"{n_params:,} params")

    done = _step(f"preprocess demo image ({_DEMO_IMAGE.name})")
    image = si._preprocess(_DEMO_IMAGE, remove_bg=True, foreground_ratio=0.65).to(device)
    if image.shape != (3, 128, 128):
        done(False, f"unexpected shape {tuple(image.shape)}")
    done()

    done = _step("forward pass (image -> Gaussians)")
    view_to_world_source, rot_transform_quats = si._source_camera(1.3)
    with torch.no_grad():
        reconstruction = model(
            image.unsqueeze(0).unsqueeze(0),
            view_to_world_source.to(device),
            rot_transform_quats.to(device),
            None,
        )
    n_gaussians = reconstruction["xyz"].shape[1]
    if n_gaussians < 1000:
        done(False, f"suspiciously few Gaussians ({n_gaussians})")
    done(detail=f"{n_gaussians} Gaussians")

    done = _step("render the reconstruction")
    reconstruction = {k: v[0].contiguous() for k, v in reconstruction.items()}
    world_view_transforms, full_proj_transforms, camera_centers = si._target_cameras(model_cfg, 1.3, 1)
    background = torch.ones(3, device=device)
    with torch.no_grad():
        render = si.render_predicted(
            reconstruction,
            world_view_transforms[0].to(device),
            full_proj_transforms[0].to(device),
            camera_centers[0].to(device),
            background,
            model_cfg,
            focals_pixels=None,
        )["render"]
    if not torch.isfinite(render).all():
        done(False, "render contains NaN/Inf")
    if render.std().item() < 1e-4:
        done(False, f"render looks blank (std={render.std().item():.2e})")
    done(detail=f"{tuple(render.shape)}, std={render.std().item():.3f}")

    print("\nSetup OK -- splatter-image inference is working end to end.")


if __name__ == "__main__":
    main()
