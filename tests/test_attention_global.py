"""diffusion.attention global: every block attends globally, the conditioning stays per block position."""
import torch

import gs_diffusion as gd
from tests.test_token_drop import FEAT, L, P, PS, inputs, make_head, rel


def test_default_pattern_unchanged_and_global_modes():
  t = gd.DiTTrunk(64, L, 9, 2)
  assert [b.mode for b in t.blocks] == ["frame", "ray", "global"] * 3 and t.cond == [b.mode for b in t.blocks]
  g = gd.DiTTrunk(64, L, 9, 2, attention="global")
  assert [b.mode for b in g.blocks] == ["global"] * 9 and g.cond == ["frame", "ray", "global"] * 3
  assert set(t.state_dict()) == set(g.state_dict())                            # same parameters: checkpoints load either way


def test_global_equals_frame_ray_global_with_the_modes_swapped():
  """The global trunk is the default trunk with every block's attention set to global (same FiLM/AdaLN per position), dense and
  packed."""
  head = make_head()
  torch.manual_seed(1)
  ref = gd.DiffusionHead(FEAT, L, patch_size=PS, attention="global")
  ref.load_state_dict(head.state_dict())
  for blk in head.trunk.blocks:
    blk.mode = "global"
  feats, x_net, t = inputs(head, 2)
  keep = torch.rand(2, L, P, generator=torch.Generator().manual_seed(0)) < 0.5
  keep[:, 0, 0] = True
  with torch.no_grad():
    assert rel(ref(feats, x_net, t), head(feats, x_net, t)) < 1e-6
    assert rel(ref(feats, x_net, t, gd.PackIndex(keep)), head(feats, x_net, t, gd.PackIndex(keep))) < 1e-6
    packed = ref(feats, x_net, t, gd.PackIndex(torch.ones(2, L, P, dtype=torch.bool)))
    assert rel(packed, ref(feats, x_net, t)) < 1e-6
