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
