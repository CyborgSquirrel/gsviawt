#!/bin/bash
#SBATCH --partition=rtx_5090
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=fin_anch
# Checks a root of scripts/fit_anchored_cluster.sbatch after its array. Nothing to finalize or convert any more: fit_anchored.py
# writes finalized sparse grids into OUT/grids_sparse/ itself. Counts: sparse grids (expected N_EXPECT, default 100 meshes x 45
# views) in OUT (and SHARED, when set), leftover *.tmp files, meshes without regression.json; then the grids open as one dataset
# (gs_diffusion.build_grid_index: one source h5). Exit 1 when anything is off. (Roots fitted with save_dense=true also hold the
# dense run dirs: finalize_fit_dataset.py collects those into grids/ if wanted; never convert_grids_sparse.py them into
# grids_sparse/, it is there already.)
# Once a shard's grids are complete: pack them into one file (pack_grids.py grids_dir=OUT/grids_sparse out=OUT/grids.h5; the
# grids_sparse files stay as they are).
#   sbatch --dependency=afterany:<array> --export=ALL,CODE=$HOME/gsviawt_<sha>,OUT=<root> .../finalize_anchored_cluster.sbatch
set -u
source ~/env.sh
cd "$CODE"
N_EXPECT=${N_EXPECT:-4500}
SHARED=${SHARED:-}
ok=1
for root in "$OUT" ${SHARED:+"$SHARED"}; do
  n=$(find "$root/grids_sparse" -maxdepth 1 -name 'mesh*_view*.h5' | wc -l)
  tmp=$(find "$root/grids_sparse" -maxdepth 1 -name '*.tmp' | wc -l)
  echo "$root/grids_sparse: $n grids (expected $N_EXPECT), $tmp leftover .tmp"
  [ "$n" -eq "$N_EXPECT" ] && [ "$tmp" -eq 0 ] || ok=0
  python -c "import sys, gs_diffusion as gd; i, s = gd.build_grid_index(sys.argv[1]); print(f'  {len(i)} grids, source {s}')" \
    "$root/grids_sparse" || ok=0
done
nreg=$(ls "$OUT"/_anchored/mesh*/regression.json 2>/dev/null | wc -l)
nmesh=$(ls -d "$OUT"/_anchored/mesh*/ 2>/dev/null | wc -l)
echo "regression.json: $nreg of $nmesh meshes"
[ "$nreg" -eq "$nmesh" ] || ok=0
[ $ok -eq 1 ] && echo FINALIZE_OK || { echo FINALIZE_INCOMPLETE; exit 1; }
