"""diffusion.augment.slot_shuffle_device=gpu (gs_diffusion.augment_x0_batch): augment_x0's slot shuffle in torch, on the collated
batch in the training step. test_augment's invariants on the torch path (on CUDA when there is one, else on CPU tensors), the
permutations' distribution, and the switch: cpu (the default) leaves the dataset as it was, gpu makes it skip the shuffle."""
import itertools

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import gs_diffusion as gd
from test_augment import FX, H, L, W, gaussians, scene
from test_shuffle_canonical import AUG, Inner

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def batch(scenes, device, half=False):
  """test_augment scenes -> (x0_raw (B, C, L, H, W), the batch's source dict) on device."""
  x0 = torch.from_numpy(np.stack([x for x, _ in scenes])).to(device)
  src = {"xyz_cam": torch.stack([s["xyz_cam"] for _, s in scenes]).to(device),
         "hit": torch.stack([s["hit"] for _, s in scenes]).to(device),
         "K_depth": torch.from_numpy(np.stack([s["K_depth"] for _, s in scenes])).to(device)}
  return (x0.half() if half else x0), src


def shuffle(scenes, device, seed, n=2, half=False, max_px=3.0):
  x0, src = batch(scenes, device, half)
  torch.manual_seed(seed)
  return gd.augment_x0_batch(x0, src, slot_shuffle_block=n, slot_shuffle_max_px=max_px).cpu().numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_preserves_gaussians_and_moves_within_blocks(device):
  scenes = [scene(0), scene(1)]
  for seed in range(5):
    for n in (2, 3):
      out = shuffle(scenes, device, seed, n)
      assert out.dtype == np.float32
      for (x0, src), o in zip(scenes, out):
        np.testing.assert_allclose(gaussians(o, src), gaussians(x0, src), atol=1e-5)
        hit = src["hit"].numpy()
        assert (o[:, ~hit] == 0).all()                               # nothing moved into an empty slot
        moved = ~np.isclose(o[0], x0[0]) & hit                       # opacity identifies a Gaussian
        assert moved.mean() > 0.3
        for l, y, x in zip(*np.nonzero(moved)):                      # each one came from within n - 1 slots
          ly, lx = np.nonzero(np.isclose(x0[0, l], o[0, l, y, x]))
          assert (np.abs(ly - y) <= n - 1).any() and (np.abs(lx - x) <= n - 1).any()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("levels", [1, 2])
def test_tied_keys_still_permute(monkeypatch, device, levels):
  """The permutation's random keys tie (float32 uniforms over ~1.5M blocks per batch: now and then): the position breaks the tie,
  so every block is still a permutation (no Gaussian duplicated, none lost). levels 1: every key equal; 2: two values."""
  real = torch.rand
  monkeypatch.setattr(torch, "rand", lambda *a, **k: (real(*a, **k) * levels).floor() / levels)
  scenes = [scene(0), scene(1)]
  for seed in range(3):
    for n in (2, 3):
      for (x0, src), o in zip(scenes, shuffle(scenes, device, seed, n)):
        np.testing.assert_allclose(gaussians(o, src), gaussians(x0, src), atol=1e-5)
        assert (o[:, ~src["hit"].numpy()] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_items_are_shuffled_independently(device):
  x0, src = scene(0)
  out = shuffle([(x0, src)] * 4, device, 0)
  assert len({o.tobytes() for o in out}) == 4


@pytest.mark.parametrize("device", DEVICES)
def test_half_input_reanchors_in_float32(device):
  """The batch's x0_raw is float16 in transit (and so is the output): the re-anchoring is done in float32 and rounded once, so the
  shuffled grid's means (anchor + offset) are the half grid's to within half an ulp of the offset (~1e-5 m here), as the numpy path's
  float32 shuffle + float16 transit; a re-anchoring in half precision would be off by ~1e-3 (half an ulp of the metre-sized anchors)."""
  x0, src = scene(0)
  half = x0.astype(np.float16).astype(np.float32)
  for seed in range(3):
    out = shuffle([(x0, src)], device, seed, half=True)[0]
    assert out.dtype == np.float16
    out = out.astype(np.float32)
    np.testing.assert_allclose(gaussians(out, src)[:, -3:], gaussians(half, src)[:, -3:], rtol=0, atol=2e-5)
    np.testing.assert_allclose(gaussians(out, src)[:, :-3], gaussians(half, src)[:, :-3], rtol=0, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_block_shift_covers_all_offsets(device):
  """With n = 2, slots 1 and 2 of a row are in one block only when the tiling is shifted by 1: over many items they do trade."""
  x0, src = scene(hole=False)
  traded = 0
  for seed in range(40):
    out = shuffle([(x0, src)], device, seed)[0]
    traded += np.isclose(out[0, 0, 0, 2], x0[0, 0, 0, 1])
  assert 0 < traded < 40


@pytest.mark.parametrize("device", DEVICES)
def test_depth_guard(device):
  x0, src = scene(hole=False)
  xyz = src["xyz_cam"].numpy().copy()
  xyz[0, :, :6, 2] = 20.0                                            # a depth edge down the middle of layer 0
  xyz[0, :, :6, :2] *= 10.0
  src = {**src, "xyz_cam": torch.from_numpy(xyz)}
  for seed in range(10):
    out = shuffle([(x0, src)], device, seed)[0]
    np.testing.assert_allclose(gaussians(out, src), gaussians(x0, src), atol=1e-5)
    for y in range(H):                                               # nothing crosses the edge between columns 5 and 6
      left = set(np.round(out[0, 0, y, :6], 5)) - set(np.round(x0[0, 0, :, :6].ravel(), 5))
      assert not left


@pytest.mark.parametrize("device", DEVICES)
def test_max_px_zero_leaves_off_centre_slots(device):
  """slot_shuffle_max_px 0: no slot's anchor is at its block's mean anchor (the scene's anchors are distinct): nothing moves."""
  x0, src = scene(0)
  np.testing.assert_allclose(shuffle([(x0, src)], device, 0, max_px=0.0)[0], x0, rtol=0, atol=1e-6)   # (a lone slot's offset is
                                                                     # re-anchored to itself: anchor + offset - anchor, to rounding)


def labelled():
  """A fully valid scene whose opacity names its slot (layer 0: y * W + x)."""
  x0, src = scene(hole=False)
  x0 = x0.copy()
  x0[0] = np.arange(L * H * W, dtype=np.float32).reshape(L, H, W)
  return x0, src


def block_permutation(o, n=2):
  """A shuffled labelled() grid -> (its tiling shift, the permutation of its first full block): the shift is the one whose blocks
  every Gaussian of layer 0 stayed in (the border slots outside the tiling unmoved)."""
  src = o[0, 0].astype(int)
  fits = []
  for sy, sx in itertools.product(range(n), range(n)):
    hb, wb = (H - sy) // n, (W - sx) // n
    ys, xs = np.mgrid[:H, :W]
    inside = (ys >= sy) & (ys < sy + hb * n) & (xs >= sx) & (xs < sx + wb * n)
    by, bx = (ys - sy) // n, (xs - sx) // n
    fy, fx = src // W, src % W
    ok = np.where(inside, ((fy - sy) // n == by) & ((fx - sx) // n == bx) & (fy >= sy) & (fx >= sx), src == ys * W + xs)
    if ok.all():
      fits.append((sy, sx))
  assert len(fits) == 1, fits
  sy, sx = fits[0]
  base = [(sy + i) * W + sx + j for i in range(n) for j in range(n)]
  return fits[0], tuple(base.index(v) for v in src[sy:sy + n, sx:sx + n].ravel())


def chi2(counts):
  e = counts.sum() / len(counts)
  return ((counts - e) ** 2 / e).sum()


@pytest.mark.parametrize("device", DEVICES)
def test_permutations_and_shifts_are_uniform(device):
  """Every permutation of a full 2x2 block (24) and every shift (4) about equally often, as augment_x0's (a chi-square test with
  p ~ 1e-4 thresholds)."""
  x0, src = labelled()
  draws = [block_permutation(o) for o in shuffle([(x0, src)] * 2400, device, 0)]
  perms, shifts = [p for _, p in draws], [s for s, _ in draws]
  assert chi2(np.array([perms.count(p) for p in itertools.permutations(range(4))])) < 55    # chi2(23): p(> 55) ~ 2e-4
  assert chi2(np.array([shifts.count(s) for s in itertools.product(range(2), range(2))])) < 21   # chi2(3): p(> 21) ~ 1e-4


def test_numpy_has_the_same_distribution():
  """The reference: augment_x0's own draws, read off the same way, pass the same test (the test reads what it should)."""
  x0, src = labelled()
  draws = [block_permutation(gd.augment_x0(x0, src, np.random.default_rng(seed), slot_shuffle=True, slot_shuffle_block=2))
           for seed in range(1200)]
  perms, shifts = [p for _, p in draws], [s for s, _ in draws]
  assert chi2(np.array([perms.count(p) for p in itertools.permutations(range(4))])) < 55
  assert chi2(np.array([shifts.count(s) for s in itertools.product(range(2), range(2))])) < 21


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------

def dataset(monkeypatch, **augment):
  x0, src = scene()
  monkeypatch.setattr(gd, "read_grid", lambda path: None)
  monkeypatch.setattr(gd, "encode_grid", lambda grid, source, rng=None, **enc: x0.copy())
  return gd.GridTargetDataset(Inner(src), {3: "x.h5"}, augment={**AUG, **augment}), x0


def test_cpu_is_the_default_and_unchanged(monkeypatch):
  """slot_shuffle_device cpu (or no key): the dataset shuffles, bit for bit as without the key."""
  for target in ("shuffled", "canonical"):
    items = []
    for extra in ({}, {"slot_shuffle_device": "cpu"}):
      ds, _ = dataset(monkeypatch, slot_shuffle_target=target, **extra)
      assert ds.shuffle_device == "cpu" and "slot_shuffle_device" not in ds.augment
      torch.manual_seed(0)
      items.append(ds[0])
    assert items[0].keys() == items[1].keys()
    for k in ("x0_raw", "x0_input_raw") if target == "canonical" else ("x0_raw",):
      torch.testing.assert_close(items[0][k], items[1][k], rtol=0, atol=0)


@pytest.mark.parametrize("target", ["shuffled", "canonical"])
def test_gpu_leaves_the_items_unaugmented(monkeypatch, target):
  ds, x0 = dataset(monkeypatch, slot_shuffle_target=target, slot_shuffle_device="gpu")
  assert ds.augment is None
  item = ds[0]
  assert "x0_input_raw" not in item
  np.testing.assert_array_equal(item["x0_raw"].numpy(), x0.astype(np.float16))


def test_unknown_device_raises(monkeypatch):
  with pytest.raises(ValueError, match="slot_shuffle_device"):
    dataset(monkeypatch, slot_shuffle_device="tpu")


def step_cfg(**augment):
  return OmegaConf.create({"diffusion": {"augment": {**AUG, **augment}}})


def collated(device=DEVICES[-1]):
  x0, src = batch([scene(0), scene(1)], device, half=True)
  return {"x0_raw": x0, "source": src, "batch_size": 2}


@pytest.mark.parametrize("target", ["shuffled", "canonical"])
def test_training_step_applies_it(target):
  import train_gs
  b = collated()
  torch.manual_seed(0)
  out = train_gs.gpu_slot_shuffle(step_cfg(slot_shuffle_target=target, slot_shuffle_device="gpu"), b)
  scenes = [scene(0), scene(1)]
  if target == "canonical":   # the target stays the grid as fitted; the input is the shuffled grid
    assert out["x0_raw"] is b["x0_raw"]
    shuffled = out["x0_input_raw"]
  else:
    assert "x0_input_raw" not in out
    shuffled = out["x0_raw"]
  assert shuffled.device == b["x0_raw"].device and shuffled.dtype == b["x0_raw"].dtype
  assert not torch.equal(shuffled, b["x0_raw"].float())
  for (_, src), o, x in zip(scenes, shuffled.float().cpu().numpy(), b["x0_raw"].float().cpu().numpy()):
    np.testing.assert_allclose(gaussians(o, src), gaussians(x, src), rtol=0, atol=2e-5)


@pytest.mark.parametrize("augment", [{"slot_shuffle_device": "cpu"}, {}, {"slot_shuffle": False, "slot_shuffle_device": "gpu"}])
def test_training_step_leaves_other_settings_alone(augment):
  import train_gs
  b = collated()
  assert train_gs.gpu_slot_shuffle(step_cfg(**augment), b) is b
