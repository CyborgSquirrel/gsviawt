"""What importing a library module pulls in (a fresh interpreter each): gs_diffusion (and the dataset / helper modules)
without lightning, wandb or trimesh; module without trimesh; and both still register the OmegaConf `eval` resolver the
configs need."""
import subprocess
import sys

import pytest

from conftest import ROOT

HEAVY = ("lightning", "wandb", "trimesh")


def imported(mod):
  code = (f"import sys; import {mod}; from omegaconf import OmegaConf; "
          f"print(','.join(m for m in {HEAVY!r} if m in sys.modules)); print(OmegaConf.has_resolver('eval'))")
  out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
  return set(filter(None, out[0].split(","))), out[1] == "True"


@pytest.mark.parametrize("mod", ["gs_diffusion", "gs_dataset", "gs_decoder", "cameras", "gaussians"])
def test_no_heavy_imports(mod):
  heavy, _ = imported(mod)
  assert not heavy, f"importing {mod} pulls in {sorted(heavy)}"


def test_module_without_trimesh():
  heavy, _ = imported("module")
  assert "trimesh" not in heavy


@pytest.mark.parametrize("mod", ["gs_diffusion", "module"])
def test_eval_resolver_registered(mod):
  assert imported(mod)[1]
