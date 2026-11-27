"""flatten_gaussians: max_sh_degree 0 output (every config) and the sh_rest branch (max_sh_degree > 0), which used to crash
(it packed one row of colors_flat with the whole batch's sh_rest, and kept pack's (tensor, shapes) tuple as the colours)."""
import pytest
import torch
from einops import rearrange

import train_gs

B, L, H, W = 2, 3, 4, 5


def fake(sh_degree, offset=False, seed=0):
  g = torch.Generator().manual_seed(seed)
  r = lambda c: torch.randn(B, L, c, H, W, generator=g)
  gauss = {"opacity": torch.rand(B, L, 1, H, W, generator=g), "scale": r(3).exp(), "rotation": r(4), "sh_dc": r(3)}
  if sh_degree > 0:
    gauss["sh_rest"] = r(3 * ((sh_degree + 1) ** 2 - 1))
  if offset:
    gauss["offset"] = r(3)
  xyz = torch.randn(B, L, H, W, 3, generator=g)
  hit = torch.rand(B, L, H, W, generator=g) < 0.6
  return gauss, xyz, hit


def old_flatten_degree0(b, gauss, xyz_cam, hit):
  """flatten_gaussians @ f0300b1 without its (crashing) sh_rest branch."""
  for idx in range(b):
    means_flat = rearrange(xyz_cam[idx], "l h w c -> (l h w) c")
    if "offset" in gauss:
      means_flat = means_flat + rearrange(gauss["offset"][idx], "l c h w -> (l h w) c")
    valid_flat = rearrange(hit[idx], "l h w -> (l h w)")
    opacity_flat = rearrange(gauss["opacity"][idx], "l c h w -> (l h w c)") * valid_flat.to(gauss["opacity"].dtype)
    keep = valid_flat
    yield {
      "means": means_flat[keep],
      "quats": rearrange(gauss["rotation"][idx], "l c h w -> (l h w) c")[keep],
      "scales": rearrange(gauss["scale"][idx], "l c h w -> (l h w) c")[keep],
      "opacities": opacity_flat[keep],
      "colors": rearrange(gauss["sh_dc"][idx], "l c h w -> (l h w) 1 c")[keep],
    }


@pytest.mark.parametrize("offset", [False, True])
def test_degree0_unchanged(offset):
  gauss, xyz, hit = fake(0, offset)
  for new, old in zip(train_gs.flatten_gaussians(B, gauss, xyz, hit), old_flatten_degree0(B, gauss, xyz, hit), strict=True):
    assert new.keys() == old.keys()
    for k in new:
      assert torch.equal(new[k], old[k]), k


@pytest.mark.parametrize("sh_degree", [1, 2])
def test_sh_rest(sh_degree):
  gauss, xyz, hit = fake(sh_degree)
  K = (sh_degree + 1) ** 2
  dc = list(train_gs.flatten_gaussians(B, {k: v for k, v in gauss.items() if k != "sh_rest"}, xyz, hit))
  for idx, flat in enumerate(train_gs.flatten_gaussians(B, gauss, xyz, hit)):
    n = int(hit[idx].sum())
    assert flat["colors"].shape == (n, K, 3)                                # what gsplat wants for sh_degree
    assert torch.equal(flat["colors"][:, :1], dc[idx]["colors"])             # band 0 = the degree-0 colours
    rest = rearrange(gauss["sh_rest"][idx], "l (k c) h w -> (l h w) k c", c=3)[rearrange(hit[idx], "l h w -> (l h w)")]
    assert torch.equal(flat["colors"][:, 1:], rest)                         # this item's own coefficients
    for k in ("means", "quats", "scales", "opacities"):
      assert torch.equal(flat[k], dc[idx][k])
