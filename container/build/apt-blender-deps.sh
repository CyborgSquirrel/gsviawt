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
