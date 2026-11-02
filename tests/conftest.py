"""Shared pytest setup (this file is identical on every branch that adds tests).

Tests import the repo's modules from the repo root (this directory's parent). Markers:
  cuda  needs a CUDA GPU (gsplat rendering, bf16 autocast); skipped without one
  data  needs local data (fitted grids, render files) under $GSVIAWT_TEST_DATA (default: <repo>/bla); the `data_dir`
        fixture skips the test when that directory -- or the file the test asks for -- is missing
Run: python -m pytest tests  (GSVIAWT_TEST_DATA=/app/bla when running from a git worktree)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)


def pytest_configure(config):
  config.addinivalue_line("markers", "cuda: needs a CUDA GPU")
  config.addinivalue_line("markers", "data: needs local data under $GSVIAWT_TEST_DATA (default <repo>/bla)")


def pytest_collection_modifyitems(config, items):
  import torch
  if torch.cuda.is_available():
    return
  for item in items:
    if "cuda" in item.keywords:
      item.add_marker(pytest.mark.skip(reason="needs a CUDA GPU"))


@pytest.fixture
def data_dir():
  """The local data directory; a function data_dir(*parts) -> path that skips the test when the path does not exist."""
  root = os.environ.get("GSVIAWT_TEST_DATA", os.path.join(ROOT, "bla"))

  def path(*parts):
    p = os.path.join(root, *parts)
    if not os.path.exists(p):
      pytest.skip(f"local test data missing: {p}")
    return p
  return path
