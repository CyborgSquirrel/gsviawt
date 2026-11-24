#!/bin/bash
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fit_anch
# Anchored layered 3DGS (fit_anchored.py) over a block of meshes of a render h5, one Slurm (array) task per block:
# task i fits meshes i*PER .. i*PER+PER-1 (or MESHES="1 3 7" explicitly), into OUT (fine-tuned) and OUT_shared.
# Regression mode with held-out views 45-49 (= the default val_fraction split of a 50-view mesh, train views 0-44), so
# every mesh gets _anchored/mesh<M>/regression.json (held-out PSNR over black). Extra args = Hydra overrides for every
# fit (e.g. loss.background=color). No finalize here: concurrent finalize_fit_dataset.py runs on one root race; run
# scripts/finalize_anchored_cluster.sbatch once after the array. Resumable: resubmit the same array.
#   sbatch --array=0-9%4 --time=02:00:00 --export=ALL,CODE=$HOME/gsviawt_<sha>,H5=<h5>,OUT=<root> \
#     $HOME/gsviawt_<sha>/scripts/fit_anchored_cluster.sbatch loss.background=color
set -u
source ~/env.sh
cd "$CODE"
SHARED=${OUT}_shared
PER=${PER:-10}
I=${SLURM_ARRAY_TASK_ID:-0}
MESHES=${MESHES:-$(seq $((I * PER)) $((I * PER + PER - 1)))}
mkdir -p "$OUT/_anchored" "$SHARED"
echo "=== task $I on $(hostname) $(date) meshes: $(echo $MESHES)"
FAILED=()
for m in $MESHES; do
  if python fit_anchored.py hdf5_path="$H5" mesh_index=$m output_dir="$OUT" shared_output_dir="$SHARED" \
      'regression.held_views=[45,46,47,48,49]' "$@" >> "$OUT/_anchored/mesh$m.log" 2>&1; then
    echo "done mesh $m $(date +%T)"
  else
    echo "FAILED mesh $m $(date +%T) (see $OUT/_anchored/mesh$m.log)"
    FAILED+=("$m")
  fi
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== task $I done $(date), FAILED meshes: ${FAILED[*]}"
  exit 1
fi
echo "=== task $I done $(date), no failures"
