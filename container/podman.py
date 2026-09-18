#!/usr/bin/env python3
"""Thin podman-compose wrapper for this project's docker-compose.yml.

podman-compose doesn't understand two things this project's GPU story
relies on under Docker:

  - docker-compose.yml's app-gpu `deploy.resources.reservations.devices`
    block is silently ignored (podman-compose 1.0.6 has no support for the
    compose `deploy` device-reservation schema at all).
  - Rootless podman remaps container UIDs by default, so the bind-mounted
    `.:/app` (owned by the host user) would come up root-owned and
    unwritable inside the container without --userns=keep-id.

Both are fixed the same way: podman-native flags injected via
podman-compose's --podman-run-args (which podman-compose forwards to both
`podman run` and `podman create`, i.e. covers `up`, `run`, and `start`).
This wrapper detects whether a GPU + nvidia-container-runtime are actually
present and adds --runtime accordingly, always adds --userns=keep-id (the
bind-mount fix applies regardless of GPU), sets XUID/XGID the same way
scripts/build already does, and forwards everything else to podman-compose
unchanged.

Usage: identical to podman-compose, e.g.
    container/podman.py build app-gpu
    container/podman.py up -d app-gpu
    container/podman.py exec app-gpu python3 /app/train_gs.py ...

podman-compose 1.0.6 doesn't support compose `profiles:`, so (unlike
`docker compose`) you always need to name the service explicitly -- there
is no `--profile gpu` equivalent.

Every invocation first regenerates Dockerfile.podman + container/build/*.sh
from the heredoc-based Dockerfile (see gen_podman_dockerfile.py) -- cheap pure
text processing, run unconditionally rather than trying to guess which
subcommands (build/up/run/create) might trigger an implicit build.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_podman_dockerfile  # noqa: E402

NVIDIA_RUNTIME = "/usr/bin/nvidia-container-runtime"


def gpu_available() -> bool:
    """Best-effort check: nvidia-container-runtime installed and a GPU responds."""
    if not os.path.exists(NVIDIA_RUNTIME):
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def main() -> None:
    if shutil.which("podman-compose") is None:
        print("podman.py: podman-compose not found on PATH", file=sys.stderr)
        sys.exit(1)

    gen_podman_dockerfile.generate()

    env = os.environ.copy()
    env.setdefault("XUID", str(os.getuid()))
    env.setdefault("XGID", str(os.getgid()))

    # Bind-mount ownership fix applies to every service, GPU or not.
    podman_run_args = ["--userns=keep-id"]
    if gpu_available():
        podman_run_args.insert(0, f"--runtime={NVIDIA_RUNTIME}")
    else:
        print("podman.py: no GPU detected, running without --runtime", file=sys.stderr)

    # -f docker-compose.podman.yml overrides build.dockerfile to point at the
    # generated Dockerfile.podman instead of the heredoc-based Dockerfile
    # Docker itself builds from -- see docker-compose.podman.yml.
    cmd = ["podman-compose", "-f", "docker-compose.yml", "-f", "docker-compose.podman.yml"]

    # Values start with "--" themselves, so argparse needs the `=` form
    # (`--podman-run-args value` as two argv tokens makes argparse treat the
    # value as another flag and reject it with "expected one argument").
    for arg in podman_run_args:
        cmd.append(f"--podman-run-args={arg}")
    cmd += sys.argv[1:]

    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as e:
        print(f"podman.py: failed to exec {cmd[0]}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
