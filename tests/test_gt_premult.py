"""The render h5 stores STRAIGHT alpha (rgb is the surface colour even where alpha ~ 0) while gsplat renders are premultiplied over
black: the photometric loss / training image metrics compare against train_gs.premultiplied(views, idx), and the background
recompositing on top of it equals the straight-alpha composite rgb * a + bg * (1 - a)."""
import torch

import train_gs


def soft_scene(B=2, V=3, H=16, W=16, seed=0):
  g = torch.Generator().manual_seed(seed)
  rgb = torch.rand(B, V, 3, H, W, generator=g) * 0.8 + 0.2              # straight colour, nonzero even where alpha is 0
  alpha = torch.zeros(B, V, 1, H, W)
  alpha[..., 4:12, 4:12] = 1
  alpha[..., 3, 3:13] = alpha[..., 12, 3:13] = alpha[..., 3:13, 3] = alpha[..., 3:13, 12] = 0.3   # a semi-transparent rim
  return {"rgb": rgb, "alpha": alpha}


def test_premultiplied_is_rgb_times_alpha():
  v = soft_scene()
  idx = [1, 2]
  p = train_gs.premultiplied(v, idx)
  assert p.shape == (2, 2, 3, 16, 16)
  torch.testing.assert_close(p, v["rgb"][:, idx] * v["alpha"][:, idx])
  assert float(p[..., 0, 0].abs().max()) == 0                            # no colour where nothing is covered
  assert float((v["rgb"][:, idx] - p).abs()[..., 0, 0].min()) > 0.1     # while the straight colour there is not 0


def test_perfect_render_scores_perfectly_only_against_the_premultiplied_gt():
  v = soft_scene()
  idx = [1, 2]
  render = v["rgb"][:, idx] * v["alpha"][:, idx]                          # what an exact gsplat render over black gives
  good = train_gs.image_quality(render, train_gs.premultiplied(v, idx), v["alpha"][:, idx])
  bad = train_gs.image_quality(render, v["rgb"][:, idx], v["alpha"][:, idx])
  assert float(good["psnr"].min()) > 60 and float(good["bg_l1"].max()) == 0
  assert float(bad["psnr"].max()) < 15 and float(bad["bg_l1"].min()) > 0.1


def test_background_recomposite_matches_straight_composite():
  v = soft_scene()
  idx = [1, 2]
  bg = torch.tensor([[0.9, 0.1, 0.5], [1.0, 1.0, 1.0]])                 # per volume, as train_gs's bg_colors
  a = v["alpha"][:, idx]
  ours = train_gs.premultiplied(v, idx) + bg[:, None, :, None, None] * (1 - a)
  straight = v["rgb"][:, idx] * a + bg[:, None, :, None, None] * (1 - a)
  torch.testing.assert_close(ours, straight)
