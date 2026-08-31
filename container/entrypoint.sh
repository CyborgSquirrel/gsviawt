#!/usr/bin/env bash
set -euo pipefail

# /app/scripts/wandb_login || true

[ -d "/mounts/venv" ] && {
  rm -rf /mounts/venv/*
  cp -r /home/user/venv/* /mounts/venv
}

exec "$@"
