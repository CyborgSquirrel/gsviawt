# syntax=docker/dockerfile:1.4

FROM docker.io/nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG XUID
ARG XGID

RUN --mount=type=bind,src=container/build/check-build-args.sh,dst=/tmp/check-build-args.sh \
  bash -eo pipefail /tmp/check-build-args.sh

############################################################
#                           User                           #
############################################################

RUN --mount=type=bind,src=container/build/create-user.sh,dst=/tmp/create-user.sh \
  bash -eo pipefail /tmp/create-user.sh

USER user

# Setup env
ENV HOME="/home/user"
ENV PATH="/home/user/.local/bin:$PATH"
RUN --mount=type=bind,src=container/build/user-dirs.sh,dst=/tmp/user-dirs.sh \
  bash -eo pipefail /tmp/user-dirs.sh

# Point HISTFILE at a symlink into a directory instead of bind-mounting the
# file directly: if a bind-mounted file's host source doesn't exist yet,
# Docker creates it as a directory (root-owned) instead, silently breaking
# history. A directory target has no such ambiguity, and bash creates the
# history file inside it on first write.
RUN --mount=type=bind,src=container/build/bash-history.sh,dst=/tmp/bash-history.sh \
  bash -eo pipefail /tmp/bash-history.sh

# Flush bash history after every command instead of only on clean shell
# exit: the entrypoint execs bash as PID 1, so a SIGTERM (e.g. `docker
# compose down`) skips the normal exit hook and would otherwise drop
# whatever history hasn't been flushed yet.
RUN echo "PROMPT_COMMAND=\"history -a\${PROMPT_COMMAND:+; \$PROMPT_COMMAND}\"" >> /home/user/.bashrc

USER root

############################################################
#                           Misc                           #
############################################################

# Update apt
RUN \
  --mount=type=cache,dst=/var/cache/apt,sharing=locked,id=apt-cache \
  --mount=type=cache,dst=/var/lib/apt,sharing=locked,id=apt-lib \
  apt-get update

# System deps
RUN \
  --mount=type=cache,dst=/var/cache/apt,sharing=locked,id=apt-cache \
  --mount=type=cache,dst=/var/lib/apt,sharing=locked,id=apt-lib \
  --mount=type=bind,src=container/build/apt-system-deps.sh,dst=/tmp/apt-system-deps.sh \
  bash -eo pipefail /tmp/apt-system-deps.sh

############################################################
#                         Blender                          #
############################################################

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

ENV BLENDER_USER_PYTHON=/home/user/.local/lib/blender-python
ARG BLENDER_VERSION=4.2.3

# Runtime libs Blender's headless build needs: GL/EGL userspace, X11 stubs
# (some Blender code paths still probe for them even with no display),
# audio stubs, and fonts (UI text rendering during import, even headless).
RUN \
  --mount=type=cache,dst=/var/cache/apt,sharing=locked,id=apt-cache \
  --mount=type=cache,dst=/var/lib/apt,sharing=locked,id=apt-lib \
  --mount=type=bind,src=container/build/apt-blender-deps.sh,dst=/tmp/apt-blender-deps.sh \
  bash -eo pipefail /tmp/apt-blender-deps.sh

# Official tarball, not apt's `blender` package: apt's build is stale and
# frequently lacks CUDA/OptiX device support and a working EGL path.
RUN \
  --mount=type=cache,target=/var/cache/blender-dl,id=blender-dl \
  --mount=type=bind,src=container/build/install-blender.sh,dst=/tmp/install-blender.sh \
  bash -eo pipefail /tmp/install-blender.sh

# Install packages for Blender's Python.
RUN \
  --mount=type=cache,target=/home/user/.cache/pip,id=pip \
  --mount=type=bind,src=container/build/install-blender-python-pkgs.sh,dst=/tmp/install-blender-python-pkgs.sh \
  bash -eo pipefail /tmp/install-blender-python-pkgs.sh

############################################################
#                            uv                            #
############################################################

USER user
WORKDIR /app

# NOTE: Blender only available either for Python 3.11, or 3.13, they jump over
# 3.12 for some reason...
ARG PYTHON_VERSION=3.13

# Install uv
RUN --mount=type=bind,src=container/uv/install.sh,dst=/tmp/install_uv.sh /tmp/install_uv.sh
ENV UV_PYTHON_CACHE_DIR=/home/user/.cache/uv

# Create virtual environment
RUN --mount=type=cache,uid=$XUID,gid=$XGID,dst=$UV_PYTHON_CACHE_DIR,id=uv \
  uv venv --python $PYTHON_VERSION /home/user/venv

# Activate venv by modifying PATH
ENV UV_PROJECT_ENVIRONMENT=/home/user/venv
ENV VIRTUAL_ENV=/home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# Install world tracing's dependencies only (no source yet, so nothing to
# build/register as editable). Cache-friendly: this layer only depends on
# pyproject.toml, so unrelated source edits don't invalidate it.
COPY --chown=$XUID:$XGID world-tracing/pyproject.toml world-tracing/pyproject.toml
RUN \
  --mount=type=cache,uid=$XUID,gid=$XGID,dst=$UV_PYTHON_CACHE_DIR,id=uv \
  --mount=type=bind,src=container/build/uv-sync-wt-deps.sh,dst=/tmp/uv-sync-wt-deps.sh \
  bash -eo pipefail /tmp/uv-sync-wt-deps.sh

# Install other packages
COPY --chown=$XUID:$XGID requirements.txt requirements.txt
RUN \
  --mount=type=cache,uid=$XUID,gid=$XGID,dst=$UV_PYTHON_CACHE_DIR,id=uv \
  --mount=type=bind,src=container/build/uv-install-requirements.sh,dst=/tmp/uv-install-requirements.sh \
  bash -eo pipefail /tmp/uv-install-requirements.sh

# gsplat: point torch.utils.cpp_extension at the pip CUDA toolchain.
ENV CUDA_HOME="/home/user/venv/lib/python3.13/site-packages/nvidia/cu13"
ENV PATH="/home/user/venv/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH"

# Pre-compile gsplat's CUDA kernels into the image so no container ever
# JIT-compiles them on first use (that's a ~2-4 min stall, and ~/.cache isn't
# a mounted volume so it would otherwise recur on every `compose down && up`).
# The build host has no GPU, so the target archs must be explicit: 8.6 = RTX
# 3080 (local dev), 8.9 = L4 (Modal); +PTX lets newer cards JIT from PTX. The
# compiled .so lands in ~/.cache/torch_extensions and is loaded as-is at
# runtime. This layer only rebuilds when requirements.txt changes.
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"
RUN python -c "import gsplat; print('gsplat', gsplat.__version__, '- CUDA kernels prebuilt')"

# Copy everything (this is the only place world-tracing's actual source
# lands in the image)
COPY --chown=$XUID:$XGID . .

# Register world tracing itself now that its source exists. Cheap: all of
# its dependencies were already installed above, so this just builds/links
# the `wt` package. Must come after `COPY . .`, not before — `uv pip install
# -e` run against a source-less directory bakes an editable-install finder
# with zero packages, so `import wt` fails forever even though uv reports it
# as installed.
RUN \
  --mount=type=cache,uid=$XUID,gid=$XGID,dst=$UV_PYTHON_CACHE_DIR,id=uv \
  --mount=type=bind,src=container/build/uv-register-wt.sh,dst=/tmp/uv-register-wt.sh \
  bash -eo pipefail /tmp/uv-register-wt.sh

# Set entrypoint
ENTRYPOINT ["/app/container/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
