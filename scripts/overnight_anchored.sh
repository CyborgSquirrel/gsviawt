#!/bin/bash
# Anchored layered 3DGS over every mesh of a render h5 (fit_anchored.py, one process per mesh), then finalize_fit_dataset.py
# on both output roots. Resumable: rerun the same command (finished meshes / views / stages are skipped, finalized grids too).
# Run inside the GPU container from the repo root, e.g.
#   scripts/overnight_anchored.sh /app/bla/new90/trellis_new90_50views_eevee_504.h5 /app/bla/new90/anchored
# Extra arguments are passed on to every fit_anchored.py call (Hydra overrides, e.g. finetune.steps=1000).
set -u
H5=$1; OUT=$2; shift 2
SHARED=${OUT}_shared
PY=${PY:-/home/user/venv/bin/python}
mkdir -p "$OUT/_anchored"
exec >> "$OUT/overnight.log" 2>&1
echo "=== start $(date)"
MESHES=$($PY -c "import h5py, numpy as np; print(' '.join(map(str, np.unique(h5py.File('$H5', 'r')['mesh_index'][:]))))")
for m in $MESHES; do
  $PY fit_anchored.py hdf5_path="$H5" mesh_index=$m output_dir="$OUT" shared_output_dir="$SHARED" "$@" \
    >> "$OUT/_anchored/mesh$m.log" 2>&1 || echo "FAILED mesh $m"
  echo "done mesh $m $(date +%T)"
done
$PY finalize_fit_dataset.py root="$OUT"
$PY finalize_fit_dataset.py root="$SHARED"
echo "=== all done $(date)"
