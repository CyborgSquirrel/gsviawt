"""Per-Gaussian shares of a render's squared error.

Alpha compositing makes the render at pixel p  C_p = sum_i w_ip c_i + T_p bg,  with w_ip = alpha_ip T_ip (Gaussian i's blending
weight there) and sum_i w_ip + T_p = 1. So the error e_p = C_p - GT_p splits exactly over the Gaussians and the background,
  e_p = sum_i w_ip (c_i - GT_p) + T_p (bg - GT_p),
and so does the squared error:
  sum_p |e_p|^2 = sum_i share_i + B,   share_i = sum_p e_p . w_ip (c_i - GT_p),   B = sum_p e_p . T_p (bg - GT_p).
share_i > 0: Gaussian i pulls the render away from the GT (a floater, a wrong colour); < 0: it pulls it towards the GT (the
background term or other Gaussians carry the corresponding error). B: what the uncovered part of the frame (T_p) gets wrong.

The w_ip are never materialised: gsplat's backward gives, for an upstream gradient g_p on the rendered channels, the gradient
d/dc_i = sum_p w_ip g_p w.r.t. a Gaussian's (precomputed, sh_degree=None) colour channels. One render of 8 channels
[c_i, 1, c_i, 1] with upstream [e_p, -(e_p . GT_p)] masked to the GT silhouette (GT alpha > 0.5) in the first four and to the rest
of the frame in the last four gives share_in_i = c_i . G[0:3] + G[3] and share_out_i = c_i . G[4:7] + G[7] (a sum over pixels, so
the split is exact); share = share_in + share_out.

The shares are of the UNCLAMPED render (the identity needs it); the project's renderers clamp to [0, 1], which only differs where
a colour is > 1 (SH can give that): total_sq_error_clamped reports the clamped render's error alongside.

Rasterization: gsplat.rasterization with eval_diffusion.render's settings (packed=True, gsplat's defaults otherwise: near 0.01,
far 1e10, eps2d 0.3, classic mode), one view per call; raster_kwargs override them. Colours are the final per-Gaussian RGB the
renderer blends: sh_dc_to_rgb for SH degree-0 coefficients (gsplat's own map), the grid files' gaussian_colors as they are.
"""
import gsplat
import torch

from gaussians import SH_C0


def sh_dc_to_rgb(sh_dc):
  """(N, 3) or (N, 1, 3) degree-0 SH coefficients -> (N, 3) RGB, as gsplat.rasterization(sh_degree=0) blends them."""
  return torch.clamp_min(SH_C0 * sh_dc.reshape(-1, 3) + 0.5, 0.0)


def render(means, quats, scales, opacities, colors, viewmats, Ks, width, height, **raster_kwargs):
  """(rgb (V, H, W, C) un-composited (over black), alpha (V, H, W, 1)) with error_shares' rasterization settings; colors
  (N, C) any number of channels."""
  kw = dict(packed=True, render_mode="RGB") | raster_kwargs
  rgb, alpha, _ = gsplat.rasterization(means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
                                       viewmats=viewmats, Ks=Ks, width=int(width), height=int(height), sh_degree=None, **kw)
  return rgb, alpha


def composite_gt(gt_rgb, gt_alpha, bg):
  """Straight-alpha GT (V, H, W, 3) / (V, H, W, 1) over bg: what a perfect render over bg shows (the premultiplied-GT convention)."""
  return gt_rgb * gt_alpha + (1 - gt_alpha) * bg


def _bg_per_view(bg, v, device):
  bg = torch.as_tensor(bg, dtype=torch.float32, device=device)
  return bg.expand(v, 3) if bg.dim() == 1 else bg


def error_shares(means, quats, scales, opacities, colors, viewmats, Ks, width, height, gt_rgb, gt_alpha, bg, **raster_kwargs):
  """means (N, 3), quats (N, 4), scales (N, 3), opacities (N,), colors (N, 3) final RGB; viewmats (V, 4, 4), Ks (V, 3, 3);
  gt_rgb (V, H, W, 3) straight colour in [0, 1], gt_alpha (V, H, W) or (V, H, W, 1); bg (3,) or (V, 3), the colour the render
  and the GT are composited over. -> dict, float64 on the inputs' device:
    share, share_in, share_out   (N,) summed over the views (share = share_in + share_out; in: GT alpha > 0.5)
    share_views                  (V, N) the per-view shares
    background, background_in, background_out   (V,) B per view (and its split)
    total_sq_error, total_sq_error_in, total_sq_error_out, total_sq_error_clamped   (V,) sum_p |e_p|^2 per view
  Per view: share_views[v].sum() + background[v] == total_sq_error[v] up to float error."""
  v_n = viewmats.shape[0]
  dev = means.device
  gt_alpha = gt_alpha.reshape(*gt_rgb.shape[:3], 1).float()
  bgs = _bg_per_view(bg, v_n, dev)
  ones = torch.ones_like(colors[:, :1])
  col8 = torch.cat([colors, ones, colors, ones], 1).float().detach().requires_grad_(True)
  share_v = []
  share_in = torch.zeros(len(means), dtype=torch.float64, device=dev)
  share_out = torch.zeros_like(share_in)
  keys = ("background", "background_in", "background_out", "total_sq_error", "total_sq_error_in", "total_sq_error_out",
          "total_sq_error_clamped")
  per_view = {k: [] for k in keys}
  for v in range(v_n):
    rgb8, alpha = render(means, quats, scales, opacities, col8, viewmats[v:v + 1], Ks[v:v + 1], width, height, **raster_kwargs)
    t = 1 - alpha[0].detach()                                        # (H, W, 1) transmittance left for the background
    b = bgs[v]
    gt = composite_gt(gt_rgb[v].float(), gt_alpha[v], b)
    pred = rgb8[0, ..., :3].detach() + t * b
    e = pred - gt
    inside = (gt_alpha[v] > 0.5).float()
    e_gt = (e * gt).sum(-1, keepdim=True)
    up = torch.cat([e * inside, -e_gt * inside, e * (1 - inside), -e_gt * (1 - inside)], -1)
    (g,) = torch.autograd.grad(rgb8[0], col8, grad_outputs=up)
    g, c = g.double(), colors.double()
    s_in = (c * g[:, 0:3]).sum(1) + g[:, 3]
    s_out = (c * g[:, 4:7]).sum(1) + g[:, 7]
    share_in += s_in
    share_out += s_out
    share_v.append(s_in + s_out)
    bterm = (e.double() * t.double() * (b - gt).double()).sum(-1)    # (H, W)
    sq = e.double().square().sum(-1)
    m = inside[..., 0].double()
    clamped = (rgb8[0, ..., :3].detach().clamp(0, 1) + t * b).clamp(0, 1) - gt
    for k, val in (("background", bterm.sum()), ("background_in", (bterm * m).sum()), ("background_out", (bterm * (1 - m)).sum()),
                   ("total_sq_error", sq.sum()), ("total_sq_error_in", (sq * m).sum()),
                   ("total_sq_error_out", (sq * (1 - m)).sum()), ("total_sq_error_clamped", clamped.double().square().sum())):
      per_view[k].append(val)
  out = {k: torch.stack(val) for k, val in per_view.items()}
  return out | {"share": share_in + share_out, "share_in": share_in, "share_out": share_out, "share_views": torch.stack(share_v)}
