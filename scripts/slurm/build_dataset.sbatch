#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --partition=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/hdd/gsviawt_dataset/slurm-build_dataset-%j.out
# The dataset builder as ONE long-running, requeueable job in the preemptible `low` partition (build_dataset.py; plan in
# bla/plan_dataset_builder.md). Preemption = SIGTERM -> scripts/gpu-run stops the builder's process group in the container
# (TERM, then KILL) -> requeue; on restart the builder continues from its files (SLURM_RESTART_COUNT is logged). A builder that
# stops on an error (e.g. a fit error) exits non-zero: the job ends FAILED and is not requeued -- fix, then sbatch again.
#   mkdir -p /mnt/hdd/gsviawt_dataset && sbatch scripts/slurm/build_dataset.sbatch [hydra overrides...]
# The code: the checkout sbatch was run from (submit from the repo root or a worktree root), or CODE=<host checkout>;
# HOST_REPO / CTR_REPO: the repo's host path and its mount in the container (gsviawt-app-gpu-1: /app).
set -u
HOST_REPO=${HOST_REPO:-/home/student/gsviawt}
CTR_REPO=${CTR_REPO:-/app}
CODE=${CODE:-${SLURM_SUBMIT_DIR:-$HOST_REPO}}
case "$CODE" in
  "$HOST_REPO"*) ;;
  *) echo "CODE=$CODE is not under $HOST_REPO (the container sees only that)"; exit 2 ;;
esac
[ -f "$CODE/build_dataset.py" ] || { echo "no build_dataset.py in $CODE (submit from the repo root or set CODE)"; exit 2; }
WD=$CTR_REPO${CODE#$HOST_REPO}
echo "=== build_dataset job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $(date), code $CODE ($WD)"
exec "$CODE/scripts/gpu-run" -w "$WD" python build_dataset.py "$@"
