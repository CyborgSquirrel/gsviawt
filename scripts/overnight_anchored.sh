#!/bin/bash
# Anchored layered 3DGS over every mesh of a render h5 (fit_anchored.py, one process per mesh); after each mesh,
# finalize_fit_dataset.py on both output roots, so finished meshes are collected even if the run is stopped early.
# Resumable: rerun the same command (finished meshes / views / stages are skipped, finalized grids too).
# Run inside the GPU container, e.g.
#   scripts/overnight_anchored.sh /app/bla/new90/trellis_new90_50views_eevee_504.h5 /app/bla/new90/anchored
# OUT must not be a fit_3dgs.py root (fit_anchored.py refuses roots holding non-anchored fits).
# Extra arguments are passed on to every fit_anchored.py call (Hydra overrides, e.g. finetune.steps=1000).
set -u
cd "$(dirname "$0")/.."
H5=$1; OUT=$2; shift 2
SHARED=${OUT}_shared
PY=${PY:-/home/user/venv/bin/python}
mkdir -p "$OUT/_anchored"
exec >> "$OUT/overnight.log" 2>&1
echo "=== start $(date) ($(git rev-parse --short HEAD 2>/dev/null))"
MESHES=$($PY -c "import h5py, numpy as np; print(' '.join(map(str, np.unique(h5py.File('$H5', 'r')['mesh_index'][:]))))")
FAILED=()
for m in $MESHES; do
  if $PY fit_anchored.py hdf5_path="$H5" mesh_index=$m output_dir="$OUT" shared_output_dir="$SHARED" "$@" \
      >> "$OUT/_anchored/mesh$m.log" 2>&1; then
    echo "done mesh $m $(date +%T)"
  else
    echo "FAILED mesh $m $(date +%T) (see $OUT/_anchored/mesh$m.log)"
    FAILED+=("$m")
  fi
  for root in "$OUT" "$SHARED"; do
    $PY finalize_fit_dataset.py root="$root" >> "$OUT/_anchored/finalize.log" 2>&1 || echo "FAILED finalize $root after mesh $m"
  done
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== all done $(date), FAILED meshes: ${FAILED[*]}"
  exit 1
fi
echo "=== all done $(date), no failures"
