pkgs=(
  # render_objaverse.py runs the whole render inside Blender and reads
  # our Hydra config + writes the h5 from there.
  h5py
  hydra-core
)
BLENDER_MAJOR="${BLENDER_VERSION%.*}"
"/opt/blender/$BLENDER_MAJOR/python/bin/python3.11" \
  -m pip install \
  --target=$BLENDER_USER_PYTHON \
  "${pkgs[@]}"
# h5py pulls in numpy; Blender ships its own and sitecustomize.py (below)
# appends BLENDER_USER_PYTHON *after* Blender's site-packages so Blender's
# numpy wins -- but drop the duplicate so it can't shadow anything.
rm -rf "$BLENDER_USER_PYTHON"/numpy "$BLENDER_USER_PYTHON"/numpy-*.dist-info

# Patch Blender's bundled Python itself so BLENDER_USER_PYTHON lands on
# sys.path for every script it runs, not just ones that remember to do it:
# sitecustomize.py is auto-imported by the `site` module on interpreter
# startup whenever it's importable, which it is once dropped into Blender's
# own site-packages.
cat > "/opt/blender/$BLENDER_MAJOR/python/lib/python3.11/site-packages/sitecustomize.py" <<'PYEOF'
import os
import sys

_extra = os.environ.get("BLENDER_USER_PYTHON", "")
if _extra.strip():
  sys.path.append(_extra)  # append: Blender's own numpy still wins
PYEOF
