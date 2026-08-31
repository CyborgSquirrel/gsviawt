#!/usr/bin/env bash
# Wrapper script to run pylsp inside the Docker container
# This allows Helix (or any editor) to use the LSP server from the container

cd "$(dirname "$0")"

# Use docker compose exec to run pylsp in the running container
# The -T flag disables TTY allocation (required for LSP communication)
docker compose exec -T app-cpu pylsp "$@"
