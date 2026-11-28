# Local Slurm with preemption for the 3080 workstation (SL12)

Purpose: the dataset builder (`build_dataset.py`) runs as one requeueable job in partition `low`. Any job in partition `high`
that needs the GPU preempts it at once (SIGTERM, then SIGKILL after `GraceTime`), runs, and when the GPU is free again Slurm
restarts the builder, which continues from its files. Nobody has to un-pause anything. See `bla/plan_dataset_builder.md`
("Preemption via local Slurm").

**Status: untested.** The files were written without root and were never installed. `scripts/gpu-run` and the builder's
kill and resume were tested without Slurm (see the plan). Check every step below on the machine.

## Files

- `slurm.conf`: one node, `SL12`, with 16 CPUs, about 60 GB of RAM and `gpu:1`.
  - Partition `low` has `PriorityTier=1`, `PreemptMode=REQUEUE` and `GraceTime=15`.
  - Partition `high` has `PriorityTier=2`.
  - Preemption uses `preempt/partition_prio` and `select/cons_tres`.
- `gres.conf`: `/dev/nvidia0` as `gpu`.
- `build_dataset.sbatch`: the builder job. It runs in `low` with `--gres=gpu:1 --cpus-per-task=8 --requeue` and calls
  `scripts/gpu-run`.
- `../gpu-run`: the Docker-aware wrapper. Everything that uses the GPU should run through it, inside a Slurm job.

## Install (needs sudo; Ubuntu 24.04, slurm-wlm 23.11)

```bash
sudo apt install -y slurm-wlm munge
# munge: the package creates /etc/munge/munge.key; if it is missing:  sudo /usr/sbin/mungekey --create && sudo chown munge: /etc/munge/munge.key
sudo systemctl enable --now munge

# check the hardware line: compare `slurmd -C` with the NodeName line in slurm.conf (CPUs / RealMemory) and adjust
slurmd -C

cd /home/student/gsviawt          # a checkout containing scripts/slurm (the dataset-builder branch)
sudo cp scripts/slurm/slurm.conf scripts/slurm/gres.conf /etc/slurm/
sudo mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
sudo chown slurm: /var/spool/slurmctld /var/log/slurm
sudo systemctl enable --now slurmctld slurmd

sinfo                              # low* and high, node SL12 idle
scontrol show node SL12 | grep -i gres   # Gres=gpu:1
```

If the node shows `drain` or `down`, run `sudo scontrol update nodename=SL12 state=resume` and read
`/var/log/slurm/slurmd.log`. A RealMemory that is larger than the machine has is the usual cause.

The job user needs Docker access, because `gpu-run` calls `docker exec`. `student` is in the `docker` group.

## Use

Start the builder once. It requeues itself after every preemption.

```bash
mkdir -p /mnt/hdd/gsviawt_dataset
cd /home/student/gsviawt           # submit from the checkout to run (the repo root or a worktree root)
sbatch scripts/slurm/build_dataset.sbatch            # extra args = Hydra overrides of conf/build_dataset.yaml
squeue; tail -f /mnt/hdd/gsviawt_dataset/logs/build.log
docker exec -w /app gsviawt-app-gpu-1 python build_dataset.py mode=status   # progress, bad meshes, ETA
```

Run every other GPU task in `high` through gpu-run. Such a task preempts the builder if it needs the GPU:

```bash
srun -p high --gres=gpu:1 scripts/gpu-run -w /app python train_gs.py ...
srun -p high --gres=gpu:1 scripts/gpu-run -w /app/.claude/worktrees/<wt> -e GSVIAWT_TEST_DATA python -m pytest -q tests/
sbatch -p high --gres=gpu:1 --wrap "scripts/gpu-run -w /app python eval_diffusion.py ..."
```

Several `high` tasks queue and run in order. When the last one ends, the requeued builder job starts again by itself.
Work that bypasses Slurm, such as a plain `docker exec` on the GPU, is invisible to it. The builder then shares the GPU with
that work, and either side may run out of memory.

## The downloader

The downloader is CPU only and is not a Slurm job. The builder starts it as a child process when none is running. That child
stops when the builder is preempted.

To keep prefetching while the builder is preempted, run a downloader of your own in the container. It holds a lock, so only
one runs, and it exits when the target is reached:

```bash
docker exec -d -w /app gsviawt-app-gpu-1 python build_dataset.py mode=download   # same overrides as the builder, if any
```

## How preemption reaches the container

Processes started by `docker exec` live in the container. They are not in the Slurm job's process tree, so Slurm can't kill
them, and killing the `docker exec` client does not stop them.

`gpu-run` starts the command as the leader of its own process group inside the container and records the group id in the
container at `/tmp/gpu-run/<id>.pgid`. On SIGTERM or SIGINT it sends TERM to the whole group in the container. If the group
is still alive after `-g` seconds (default 8, which must stay below `GraceTime`), it sends KILL, then waits until the group is
gone.

SIGKILL can't be trapped. If gpu-run itself is SIGKILLed first, which is what Slurm does after `GraceTime`, the container
processes keep running. `GraceTime=15` leaves gpu-run enough time.

If processes are ever left over, the builder copes with them:
- It refuses to start a second instance, using the `out_root/.builder.lock` flock.
- Its render stage refuses while Blender workers of its queue are alive.
- Blender workers die with their launcher (`PR_SET_PDEATHSIG`).

To clean up by hand:
`docker exec gsviawt-app-gpu-1 pkill -f build_dataset.py`, then `nvidia-smi`.
