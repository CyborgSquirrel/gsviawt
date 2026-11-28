"""error_shares: the per-Gaussian shares of the squared error add up to it exactly (with the background term), match the shares
from explicitly rendered blending weights, vanish for a perfect render and land on the Gaussian (or the background) that is wrong."""
import gsplat
import pytest
import torch

from error_shares import error_shares, render, sh_dc_to_rgb

pytestmark = pytest.mark.cuda
H = W = 32


def camera(v=1):
  """v cameras at the origin looking down +Z (OpenCV), focal 32 px, for a W x H image; the views after the first slightly turned."""
  vm = torch.eye(4).repeat(v, 1, 1)
  for i in range(1, v):
    a = torch.tensor(0.15 * i)
    vm[i, 0, 0], vm[i, 0, 2], vm[i, 2, 0], vm[i, 2, 2] = a.cos(), a.sin(), -a.sin(), a.cos()
    vm[i, 0, 3] = -0.3 * a.sin()
  K = torch.tensor([[32.0, 0, W / 2], [0, 32.0, H / 2], [0, 0, 1]]).repeat(v, 1, 1)
  return vm.cuda(), K.cuda()


def scene(n, seed, spread=0.4, scale=(0.05, 0.2), opacity=(0.3, 1.0)):
  """n random overlapping Gaussians around (0, 0, 3): means, quats, scales, opacities, colors."""
  g = torch.Generator().manual_seed(seed)
  r = lambda *s: torch.rand(*s, generator=g)
  means = (r(n, 3) - 0.5) * 2 * spread + torch.tensor([0.0, 0.0, 3.0])
  quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
  scales = scale[0] + r(n, 3) * (scale[1] - scale[0])
  opac = opacity[0] + r(n) * (opacity[1] - opacity[0])
  return [t.cuda() for t in (means, quats, scales, opac, r(n, 3))]


def as_gt(g, vm, K, bg):
  """The scene's own render as straight-alpha GT: (gt_rgb, gt_alpha) with composite_gt(gt_rgb, gt_alpha, bg) == the render."""
  rgb, alpha = render(*g, vm, K, W, H)
  return rgb / alpha.clamp_min(1e-12), alpha


def shares(g, vm, K, gt_rgb, gt_alpha, bg):
  return error_shares(*g, vm, K, W, H, gt_rgb, gt_alpha, torch.tensor(bg).cuda())


def random_gt(v, seed):
  g = torch.Generator().manual_seed(seed)
  alpha = (torch.rand(v, H, W, 1, generator=g) > 0.4).float() * torch.rand(v, H, W, 1, generator=g).clamp_min(0.5)
  return torch.rand(v, H, W, 3, generator=g).cuda(), alpha.cuda()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shares_and_background_sum_to_the_squared_error(seed):
  v = 3
  vm, K = camera(v)
  g = scene(200, seed)
  out = shares(g, vm, K, *random_gt(v, seed), [0.2, 0.5, 0.9])
  total = out["total_sq_error"]
  assert float(total.min()) > 1
  torch.testing.assert_close(out["share_views"].sum(1) + out["background"], total, rtol=1e-4, atol=0)
  torch.testing.assert_close(out["share"], out["share_in"] + out["share_out"])
  torch.testing.assert_close(out["share_in"].sum() + out["background_in"].sum(), out["total_sq_error_in"].sum(), rtol=1e-4, atol=0)
  torch.testing.assert_close(out["share_out"].sum() + out["background_out"].sum(), out["total_sq_error_out"].sum(), rtol=1e-4,
                             atol=0)


def blend_weights(g, vm, K):
  """(w (H, W, N), T (H, W, 1)) of view 0: render Gaussian i with the one-hot colour e_i, so channel i of the render is w_ip."""
  n = len(g[0])
  w, alpha = render(*g[:4], torch.eye(n, device="cuda"), vm[:1], K[:1], W, H)
  return w[0], 1 - alpha[0]


def test_shares_match_explicit_blending_weights():
  vm, K = camera()
  g = scene(5, 3, spread=0.15, scale=(0.2, 0.5))
  bg = torch.tensor([0.1, 0.6, 0.3]).cuda()
  gt_rgb, gt_alpha = random_gt(1, 3)
  w, t = blend_weights(g, vm, K)
  c = g[4]
  gt = gt_rgb[0] * gt_alpha[0] + (1 - gt_alpha[0]) * bg
  e = (w @ c + t * bg) - gt
  brute = torch.einsum("hwc,hwi,ihwc->i", e, w, c[:, None, None, :] - gt[None])
  out = shares(g, vm, K, gt_rgb, gt_alpha, bg.tolist())
  assert float((w.sum(-1) > 0.5).float().mean()) > 0.1            # the Gaussians cover a good part of the frame ...
  assert float(((w > 0.05).sum(-1) >= 2).float().mean()) > 0.1     # ... overlapping
  torch.testing.assert_close(out["share"].float(), brute, rtol=1e-4, atol=1e-5)
  torch.testing.assert_close(out["background"][0].float(), (e * t * (bg - gt)).sum(), rtol=1e-4, atol=1e-5)


def test_blending_weights_and_transmittance_sum_to_one():
  vm, K = camera()
  w, t = blend_weights(scene(5, 3, spread=0.15, scale=(0.2, 0.5)), vm, K)
  torch.testing.assert_close(w.sum(-1, keepdim=True) + t, torch.ones_like(t), rtol=0, atol=1e-5)


def test_own_render_as_gt_gives_zero_shares():
  vm, K = camera(2)
  g = scene(200, 4)
  bg = [0.3, 0.3, 0.8]
  out = shares(g, vm, K, *as_gt(g, vm, K, torch.tensor(bg).cuda()), bg)
  assert float(out["total_sq_error"].max()) < 1e-8                 # float32 rounding of rgb / alpha * alpha only
  assert float(out["share"].abs().max()) < 1e-6 and float(out["background"].abs().max()) < 1e-6


def test_planted_floater_takes_the_added_error():
  vm, K = camera(2)
  g = scene(100, 5, spread=0.2)
  bg = [1.0, 1.0, 1.0]
  gt = as_gt(g, vm, K, torch.tensor(bg).cuda())
  floater = [torch.tensor(x, device="cuda")[None] for x in ([1.1, -1.1, 3.0], [1.0, 0, 0, 0], [0.1, 0.1, 0.1], 1.0, [0.1, 0.1, 0.1])]
  with_f = [torch.cat([a, f]) for a, f in zip(g, floater)]        # opaque, dark, in the empty top-right corner of the frame
  out = shares(with_f, vm, K, *gt, bg)
  total = out["total_sq_error"].sum()
  assert float(total) > 1                                          # the floater does add error
  torch.testing.assert_close(out["share"][-1], total, rtol=1e-4, atol=0)
  assert float(out["share"][:-1].abs().max()) < 1e-6 * float(total)
  assert float(out["share_in"][-1].abs()) < 1e-6 * float(total)    # all of it outside the GT silhouette


def test_recoloured_gaussian_carries_the_error():
  vm, K = camera(2)
  g = scene(100, 6, spread=0.2)
  k = 0                                                           # opaque, in front of the rest: well visible, nothing over it
  g[0][k], g[2][k], g[3][k] = torch.tensor([0.0, 0.0, 2.4]).cuda(), 0.12, 1.0
  bg = [0.0, 0.0, 0.0]
  gt = as_gt(g, vm, K, torch.tensor(bg).cuda())
  colours = g[4].clone()
  colours[k] = 1 - colours[k]
  out = shares([*g[:4], colours], vm, K, *gt, bg)
  total, s = out["total_sq_error"].sum(), out["share"]
  assert float(s[k]) > 0.8 * float(total)                        # it gets nearly all of the error ...
  others = s.clone()
  others[k] = 0
  assert float(others.abs().max()) < 0.1 * float(s[k])            # ... each other one (behind its soft edge) little
  torch.testing.assert_close(s.sum() + out["background"].sum(), total, rtol=1e-4, atol=0)


def test_uncovered_gt_goes_to_the_background_term():
  vm, K = camera()
  g = scene(50, 7, spread=0.1, scale=(0.03, 0.06))                # a small object in the middle of the frame
  bg = torch.tensor([0.2, 0.2, 0.2]).cuda()
  gt_rgb, gt_alpha = as_gt(g, vm, K, bg)
  gt_rgb, gt_alpha = gt_rgb.clone(), gt_alpha.clone()
  gt_rgb[:, :6, :6], gt_alpha[:, :6, :6] = torch.tensor([0.9, 0.1, 0.4]).cuda(), 1.0   # GT content where nothing is rendered
  out = shares(g, vm, K, gt_rgb, gt_alpha, bg.tolist())
  total = out["total_sq_error"].sum()
  assert float(total) > 1
  torch.testing.assert_close(out["background"].sum(), total, rtol=1e-4, atol=0)
  torch.testing.assert_close(out["background_in"].sum(), total, rtol=1e-4, atol=0)
  assert float(out["share"].abs().max()) < 1e-6 * float(total)


def test_sh_dc_to_rgb_is_what_gsplat_blends_at_degree_0():
  vm, K = camera(2)
  means, quats, scales, opac, _ = scene(200, 8)
  sh = torch.randn(200, 1, 3, device="cuda") * 2                  # some channels come out < 0, where gsplat clamps
  ref, _, _ = gsplat.rasterization(means, quats, scales, opac, sh, vm, K, W, H, sh_degree=0, packed=True)
  ours, _ = render(means, quats, scales, opac, sh_dc_to_rgb(sh), vm, K, W, H)
  torch.testing.assert_close(ours, ref, rtol=0, atol=1e-6)
