"""The overfitting-run knobs of GSDataModule: data.catalog_transform (a polars expression picking the volumes trained / validated on,
with the splits, pools and x0 stats left the full catalogue's), data.val_train (a validation set of the training volumes) and
data.repeat (an epoch of N passes over the training volumes). The expression helper is tested on its own; the rest builds the real
GSDataModule on local grids (data marker)."""
import tempfile

import polars as pl
import pytest
from omegaconf import OmegaConf, open_dict

import train_gs


def test_transform_keys_plain_polars():
  df = pl.DataFrame({"mesh_id": [0, 0, 1, 1, 2], "view_idx": [0, 1, 2, 3, 4], "path": ["a.h5"] * 5})
  expr = 'df.filter(pl.col("mesh_id") == 1).filter(pl.col("view_idx") == 3)'
  assert train_gs.catalog_transform_keys(df, expr) == {(1, 3)}
  assert train_gs.catalog_transform_keys(df, 'df.filter(pl.col("mesh_id") != 0)') == {(1, 2), (1, 3), (2, 4)}
  assert train_gs.catalog_transform_keys(df, "df.head(0)") == set()
  assert train_gs.catalog_transform_keys(df, "df.lazy().filter(pl.col('view_idx') < 1)") == {(0, 0)}   # a LazyFrame is collected


@pytest.mark.parametrize("expr, msg", [
  ("df.height", "must return a polars DataFrame"),
  ('df.select("mesh_id")', "lost the"),
  ("df.filter(pl.col('nope') == 1)", "failed"),
  ("None", "must return a polars DataFrame"),
])
def test_transform_errors(expr, msg):
  df = pl.DataFrame({"mesh_id": [0, 1], "view_idx": [0, 1]})
  with pytest.raises(SystemExit, match=msg):
    train_gs.catalog_transform_keys(df, expr)


def keys(ds):
  df = ds.inner.volumes.df
  return set(zip(df["mesh_id"].to_list(), df["view_idx"].to_list()))


@pytest.fixture(scope="module")
def make(data_dir):
  paths = {k: data_dir(p) for k, p in (("cfg", "eval_test/run409_config.yaml"), ("grids", "fit_dataset/grids"),
                                        ("h5", "trellis_new10_50views_eevee_504.h5"))}
  base = OmegaConf.load(paths["cfg"])
  cache = {}

  def dm(**data):
    key = tuple(sorted(data.items()))
    if key not in cache:
      c = base.copy()
      with open_dict(c):
        c.output_dir = tempfile.mkdtemp()
        c.diffusion.grids_dir, c.diffusion.source_h5 = paths["grids"], paths["h5"]
        for k, v in data.items():
          c.data[k] = v
      d = train_gs.GSDataModule(c)
      d.setup()
      cache[key] = d
    return cache[key]
  return dm


@pytest.fixture(scope="module")
def picked(make):
  """A training volume (mesh, view) of the base run and its mesh's expressions."""
  base = make()
  m, v = sorted(keys(base.train_ds))[0]
  return m, v, f'df.filter(pl.col("mesh_id") == {m}).filter(pl.col("view_idx") == {v})', f'df.filter(pl.col("mesh_id") == {m})'


@pytest.mark.data
def test_transform_keeps_exactly_the_chosen_volumes(make, picked):
  m, v, one, mesh = picked
  base, a, b = make(), make(catalog_transform=one), make(catalog_transform=mesh)
  assert keys(a.train_ds) == {(m, v)}
  assert set(a.val_datasets) <= {"val_views_strain", "val_views_sval", "val"} or not a.val_datasets
  for name, ds in a.val_datasets.items():   # only sets that hold that very volume survive
    assert name == "val" or keys(ds) == {(m, v)}
  assert keys(b.train_ds) == {k for k in keys(base.train_ds) if k[0] == m}
  for name, ds in base.val_datasets.items():
    want = {k for k in keys(ds) if k[0] == m}
    assert (keys(b.val_datasets[name]) == want) if want else name not in b.val_datasets
  assert "val_views_sval" in b.val_datasets and "val_meshes" not in b.val_datasets   # a train mesh: its held-out views stay


@pytest.mark.data
def test_transform_leaves_splits_pools_and_stats(make, picked):
  base, a, b = make(), make(catalog_transform=picked[2]), make(catalog_transform=picked[3])
  for d in (a, b):
    assert all((x == y).all() for x, y in zip(base.diffusion_stats, d.diffusion_stats))   # normalized like the full run
    assert d.train_ds.inner.pool_sets == base.train_ds.inner.pool_sets                    # the same supervision views
    for name, ds in d.val_datasets.items():
      if name == "val":   # every validation set emptied: the placeholder
        continue
      assert ds.inner.pool_sets == base.val_datasets[name].inner.pool_sets
      assert ds.seeded


@pytest.mark.data
def test_transform_without_training_volume_raises(make):
  base = make()
  val_mesh = sorted(keys(base.val_datasets["val_meshes"]))[0][0]
  with pytest.raises(SystemExit, match="leaves no training volume"):
    make(catalog_transform=f'df.filter(pl.col("mesh_id") == {val_mesh})')
  held = sorted(keys(base.val_datasets["val_views_sval"]))[0]   # a held-out view of a training mesh
  with pytest.raises(SystemExit, match="leaves no training volume"):
    make(catalog_transform=f'df.filter(pl.col("mesh_id") == {held[0]}).filter(pl.col("view_idx") == {held[1]})')


@pytest.mark.data
def test_val_train_is_the_training_volumes(make, picked):
  base, vt = make(), make(val_train=True)
  assert list(vt.val_datasets) == [*base.val_datasets, "val_train"]   # appended: the other sets keep their dataloader index
  ds = vt.val_datasets["val_train"]
  assert keys(ds) == keys(vt.train_ds) == keys(base.train_ds)
  assert ds.inner.pool_sets == vt.train_ds.inner.pool_sets and ds.seeded
  assert all(keys(vt.val_datasets[n]) == keys(base.val_datasets[n]) for n in base.val_datasets)
  one = make(catalog_transform=picked[2], val_train=True)
  assert keys(one.val_datasets["val_train"]) == {picked[:2]}


@pytest.mark.data
def test_repeat(make, picked):
  base, r1, r3 = make(), make(repeat=1), make(catalog_transform=picked[3], repeat=3)
  assert base.train_dataloader().dataset is base.train_ds and r1.train_dataloader().dataset is r1.train_ds   # the unchanged path
  dl = r3.train_dataloader()
  assert len(dl.dataset) == 3 * len(r3.train_ds) and len(r3.train_ds) == len(make(catalog_transform=picked[3]).train_ds)
  assert all(part is r3.train_ds for part in dl.dataset.datasets)   # each copy the same volumes (each access its own draws)
  with pytest.raises(SystemExit, match="data.repeat"):
    make(repeat=0)
