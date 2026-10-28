#!/usr/bin/env python3
"""Extract per-view feature volumes from a render_objaverse.py h5 with a pluggable
feature extractor and save them -- the tensors a Gaussian head is trained on.

The extractor is the `extractor` config node, a Hydra `_target_` class from
`extractors.py` (`extractor=wt`, `extractor=moge`, or your own subclass of
`FeatureExtractor`; see its docstring for what each one does). For each explicit
view in `views` it yields one or more rows of `[L, P, D]` tokens.

Output h5:
    features     (N, L, P, D)  tokens, in the dtype the extractor returned
                               (bfloat16 is stored as its uint16 bit pattern,
                               since h5py/numpy have no bf16 -- see the
                               `torch_dtype` attr). gzip, one chunk per row.
    source_h5    attr          absolute path of the render h5 the views come from
    mesh_index   (N,) int64    the source h5's `mesh_index` of each row's view
                               (which mesh it is; `mesh_paths` in the source h5)
    view_indices (N,) int64    row index into the input h5 (each view repeated once
                               per row the extractor yields for it)
    + one (N,) dataset per extra column the extractor reports. WTExtractor:
      noise_level (float32) and seed (int64; rows N = len(views) * seeds_per_view *
      len(noise levels), ordered view, then seed, then noise level; the main seed is
      in the `main_seed` attr). MoGeExtractor adds none (one [1, P, D] row per view).
    attrs: extractor (name), config_json, input, torch_dtype, features layout/dtype,
           plus whatever the extractor reports (image_size, patch_size, ...).

Config is Hydra (`conf/wt_features.yaml` + `conf/extractor/*.yaml`); run in the
container venv:

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python \\
        wt_features.py input=/app/bla/lite.h5 'views=[0,5,12]' extractor.noise_level=0.5
    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python \\
        wt_features.py input=/app/bla/lite.h5 'views=[0,5,12]' extractor=moge
"""

import json
import logging
import os
import sys

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import LazyDataset  # noqa: E402

log = logging.getLogger(__name__)


def to_h5_array(tokens):
    """-> (numpy array, torch dtype string). bfloat16 has no numpy dtype, so it
    is stored as its raw uint16 bit pattern."""
    dtype = str(tokens.dtype)
    if tokens.dtype == torch.bfloat16:
        return tokens.contiguous().view(torch.int16).cpu().numpy().view(np.uint16), dtype
    return tokens.cpu().numpy(), dtype


@hydra.main(version_base=None, config_path="conf", config_name="wt_features")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    views = [int(v) for v in cfg.views]
    extractor = hydra.utils.instantiate(cfg.extractor)
    output = cfg.output or f"{cfg.input}.{extractor.name}_features.h5"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor.load(device)

    def one_chunk_per_row(ds, *, shape, dtype):
        ds.dataset_kwargs = {"chunks": (1, *shape), **ds.dataset_kwargs}

    row_views, row_meshes, columns = [], [], {}  # per output row
    feat_dtype = None
    with (
        h5py.File(cfg.input, "r") as hf,
        h5py.File(output, "w") as out,
        LazyDataset(
            out,
            "features",
            dataset_kwargs=(
                dict(compression="gzip", compression_opts=int(cfg.compression_level))
                if int(cfg.compression_level) > 0 else {}
            ),
            init_hook=one_chunk_per_row,
        ) as features,
    ):
        for index in views:
            mesh = int(hf["mesh_index"][index])
            for row in extractor.extract_view(hf, index):
                if row_views and set(row.columns) != set(columns):
                    raise ValueError(f"extractor changed its columns mid-run: {sorted(row.columns)} vs {sorted(columns)}")
                tokens = row.tokens if cfg.feature_dtype is None else row.tokens.to(getattr(torch, cfg.feature_dtype))
                arr, torch_dtype = to_h5_array(tokens)
                features.append(arr)
                row_views.append(index)
                row_meshes.append(mesh)
                for key, value in row.columns.items():
                    columns.setdefault(key, []).append(value)
                if feat_dtype is None:
                    feat_dtype = torch_dtype
                    features.dataset.attrs["layout"] = "N, L (layers), P (patches, row-major HxW grid), D"
                    features.dataset.attrs["torch_dtype"] = torch_dtype
                    features.dataset.attrs["stored_as_uint16_bits"] = torch_dtype == "torch.bfloat16"
                log.info("view %d: tokens %s %s, %s", index, tuple(row.tokens.shape), torch_dtype, row.note)

        out.create_dataset("mesh_index", data=np.array(row_meshes, dtype=np.int64))
        out.create_dataset("view_indices", data=np.array(row_views, dtype=np.int64))
        column_dtypes = {"noise_level": np.float32, "seed": np.int64, "block": np.int64}
        for key, values in columns.items():
            out.create_dataset(key, data=np.array(values, dtype=column_dtypes.get(key)))
        out.attrs["extractor"] = extractor.name
        out.attrs["config_json"] = json.dumps(OmegaConf.to_container(cfg, resolve=True))
        out.attrs["input"] = str(cfg.input)
        out.attrs["source_h5"] = os.path.abspath(str(cfg.input))
        out.attrs["torch_dtype"] = feat_dtype
        for key, value in extractor.attrs().items():
            out.attrs[key] = value

    log.info("wrote %s", output)


if __name__ == "__main__":
    main()
