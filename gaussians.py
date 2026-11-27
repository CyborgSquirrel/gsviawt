"""Small Gaussian-parameter helpers, numpy/torch only: the SH band-0 constant, the opacity logit, the 4 quaternion
forms of a Gaussian, and the layout of the diffusion model's x0 channels.

The one home of what module.py, fit_3dgs.py and gs_diffusion.py each used to define themselves (the old names are
re-exported there: `from module import SH_C0`, `gs_diffusion.quat_forms`, `gs_diffusion.SCALE`, ... keep working).
Importing this pulls in neither lightning nor wandb.

Left where they are on purpose (not the same numbers): gs_diffusion.encode_grid's own opacity logit (clip at 1e-6, then
clamped to +-OPACITY_LOGIT_CLAMP: the cached x0 moments depend on it), anchored_3dgs.py's (no clip) and
diffusion_schedule._logit (a timestep logit, torch).
"""

import numpy as np
import torch

SH_C0 = 0.28209479177387814  # SH band-0 constant, for RGB <-> sh0 in the .ply


def logit(x, eps):
  """numpy logit of x clipped into [eps, 1 - eps]. eps has no default on purpose: the call sites use different ones
  (module.init_gaussians and fit_3dgs.py 1e-4, encode_grid 1e-6), and it changes the result."""
  x = np.clip(x, eps, 1.0 - eps)
  return np.log(x / (1.0 - x))


def quat_forms(q):
  """(N, 4) wxyz -> (N, 4 forms, 4): q times 1, i, j, k -- the same Gaussian with two of its axes flipped (a 180 degree turn about
  one of its own axes). The 4 forms are exactly the rotations that leave an ellipsoid (with fixed axis order) unchanged.
  q a numpy array or a torch tensor (the result is the same kind)."""
  stack = torch.stack if isinstance(q, torch.Tensor) else np.stack
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return stack([q, stack([-x, w, z, -y], 1), stack([-y, -z, w, x], 1), stack([-z, y, -x, w], 1)], 1)


# ---------------------------------------------------------------------------
# the diffusion model's x0 channels (gs_diffusion.py): per slot, these fields in this order
# ---------------------------------------------------------------------------

X0_FIELDS = ("opacity", "scale", "rotation", "sh_dc", "offset")
X0_DIMS = (1, 3, 4, 3, 3)
X0_CHANNELS = sum(X0_DIMS)   # 14
X0_SLICES = {name: slice(int(o), int(o) + d) for name, o, d in zip(X0_FIELDS, np.cumsum((0,) + X0_DIMS)[:-1], X0_DIMS)}
