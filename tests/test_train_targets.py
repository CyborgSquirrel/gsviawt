"""diffusion.train_targets=all: training volumes are supervised from every render of their mesh except the held-out grid views;
no validation view leaks into the training targets; validation pools, training volumes and x0 stats are unchanged.
Builds the real GSDataModule on local grids (data marker)."""
import tempfile

import pytest
from omegaconf import OmegaConf, open_dict

import train_gs

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def modules(data_dir):
  paths = {k: data_dir(p) for k, p in (("cfg", "eval_test/run409_config.yaml"), ("grids", "fit_dataset/grids"),
                                        ("h5", "trellis_new10_50views_eevee_504.h5"))}
  base = OmegaConf.load(paths["cfg"])

  def dm(tt):
    c = base.copy()
    with open_dict(c):
      c.output_dir = tempfile.mkdtemp()
      c.diffusion.grids_dir, c.diffusion.source_h5, c.diffusion.train_targets = paths["grids"], paths["h5"], tt
    d = train_gs.GSDataModule(c)
    d.setup()
    return d
  return dm("grid"), dm("all")


def test_all_targets_widen_training_pools_without_leaks(modules):
  g, a = modules
  pg, pa = g.train_ds.inner.pool_sets, a.train_ds.inner.pool_sets
  assert set(pg) == set(pa) and all(pg[m] < pa[m] for m in pg)           # strictly more targets per training mesh
  val = {(m, v) for ds in g.val_datasets.values() for m, vs in ds.inner.pool_sets.items() for v in vs}
  assert not any((m, v) in val for m, vs in pa.items() for v in vs)


def test_everything_else_unchanged(modules):
  g, a = modules
  assert all(g.val_datasets[n].inner.pool_sets == a.val_datasets[n].inner.pool_sets for n in g.val_datasets)
  cols = ["mesh_id", "view_idx"]
  assert g.train_ds.inner.volumes.df.select(cols).equals(a.train_ds.inner.volumes.df.select(cols))
  assert all((x == y).all() for x, y in zip(g.diffusion_stats, a.diffusion_stats))
