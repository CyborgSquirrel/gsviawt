# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG XUID
ARG XGID

RUN <<EOF
  test -n "$XUID" || (echo "XUID build arg is required" && exit 1)
  test -n "$XGID" || (echo "XGID build arg is required" && exit 1)
EOF

############################################################
#                           User                           #
############################################################

RUN <<EOF
  # Free up UID 1000
  userdel -r ubuntu 2>/dev/null || true

  # Create user matching host group and user
  groupadd -f -g "$XGID" user
  useradd -m -u "$XUID" -g "$XGID" -s /bin/bash user
  # chown user:user /app
EOF

USER user

# Setup env
ENV HOME="/home/user"
ENV PATH="/home/user/.local/bin:$PATH"
RUN <<EOF
  mkdir -p /home/user/.cache
  mkdir -p /home/user/.local
EOF

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
<<EOF
  pkgs=(
    # Misc
      build-essential
      python3-dev
      git
      swig
      curl
      neovim
    # Graphics
      # libgl1-mesa-glx
      libglib2.0-0
      libsm6
      libxext6
      libxrender1
      libgomp1
    # Wayland support
      libwayland-client0
      libwayland-egl1
      qtwayland5
      libqt5waylandclient5
  )
  apt-get install -y "${pkgs[@]}"
EOF

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
<<EOF
  pkgs=(
    wget xz-utils ca-certificates
    libgl1 libegl1 libglvnd0 libglx0
    libxi6 libxrender1 libxfixes3 libxkbcommon0 libsm6 libxext6 libxrandr2
    libxinerama1 libxcursor1
    libsndfile1
    libopenexr-dev
    fonts-dejavu-core
  )
  apt-get install -y --no-install-recommends "${pkgs[@]}"
EOF

# Official tarball, not apt's `blender` package: apt's build is stale and
# frequently lacks CUDA/OptiX device support and a working EGL path.
RUN \
  --mount=type=cache,target=/var/cache/blender-dl,id=blender-dl \
<<EOF
  BLENDER_MAJOR="${BLENDER_VERSION%.*}"
  BLENDER_TAR="/var/cache/blender-dl/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  if [ ! -f "$BLENDER_TAR" ]; then
    wget \
      "https://download.blender.org/release/Blender${BLENDER_MAJOR}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
      -O /tmp/blender.tar.xz
    mv /tmp/blender.tar.xz "$BLENDER_TAR"
  fi
  mkdir -p /opt/blender
  tar -xf "$BLENDER_TAR" -C /opt/blender --strip-components=1
EOF

# Install packages for Blender's Python.
RUN \
  --mount=type=cache,target=/home/user/.cache/pip,id=pip \
<<EOF
  pkgs=(
    rpyc
  )
  BLENDER_MAJOR="${BLENDER_VERSION%.*}"
  "/opt/blender/$BLENDER_MAJOR/python/bin/python3.11" \
    -m pip install \
    --target=$BLENDER_USER_PYTHON \
    "${pkgs[@]}" 
EOF

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

# Install world tracing
COPY --chown=$XUID:$XGID world-tracing/pyproject.toml world-tracing/pyproject.toml
RUN \
  --mount=type=cache,uid=$XUID,gid=$XGID,dst=$UV_PYTHON_CACHE_DIR,id=uv \
<<EOF
  cd world-tracing
  uv lock
  uv sync --inexact
  uv pip install -e '.[viz]'
EOF

# Install other packages
COPY --chown=$XUID:$XGID requirements.txt requirements.txt
RUN \
  --mount=type=cache,uid=$XUID,gid=$XGID,dst=$UV_PYTHON_CACHE_DIR,id=uv \
<<EOF
  uv pip compile requirements.txt -o requirements.lock
  uv pip install -r requirements.lock
EOF

# Copy everything
COPY --chown=$XUID:$XGID . .

# Set entrypoint
ENTRYPOINT ["/app/container/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]

