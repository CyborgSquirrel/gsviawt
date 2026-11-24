#!/bin/bash
#SBATCH --partition=rtx_5090
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=fin_anch
# finalize_fit_dataset.py on OUT and OUT_shared (one process per root, after scripts/fit_anchored_cluster.sbatch's array),
# then the counts: grids per root (expected N_EXPECT, default 100 meshes x 45 views), meshes without regression.json, and
# run dirs left unfinalized. Exit 1 when a count is off.
#   sbatch --dependency=afterany:<array> --export=ALL,CODE=$HOME/gsviawt_<sha>,OUT=<root> .../finalize_anchored_cluster.sbatch
set -u
source ~/env.sh
cd "$CODE"
N_EXPECT=${N_EXPECT:-4500}
ok=1
for root in "$OUT" "${OUT}_shared"; do
  python finalize_fit_dataset.py root="$root" || ok=0
  n=$(find "$root/grids" -maxdepth 1 -name 'mesh*_view*.h5' | wc -l)
  left=$(find "$root" -maxdepth 1 -name 'mesh*_view*' -type d | wc -l)
  echo "$root: $n grids (expected $N_EXPECT), $left run dirs left"
  [ "$n" -eq "$N_EXPECT" ] && [ "$left" -eq 0 ] || ok=0
done
nreg=$(ls "$OUT"/_anchored/mesh*/regression.json 2>/dev/null | wc -l)
echo "regression.json: $nreg meshes"
[ $ok -eq 1 ] && echo FINALIZE_OK || { echo FINALIZE_INCOMPLETE; exit 1; }
