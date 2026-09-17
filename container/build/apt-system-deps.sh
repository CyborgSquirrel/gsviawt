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
