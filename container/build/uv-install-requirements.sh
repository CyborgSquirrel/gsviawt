uv pip compile requirements.txt -o requirements.lock
uv pip install -r requirements.lock
# gsplat's JIT link step passes -lcudart; the pip CUDA wheel ships only the
# versioned libcudart.so.13, so add the dev symlink it expects. (fit_gsplat
# also does this at runtime, for anyone who pip-installs into an existing env.)
cudalib="/home/user/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cu13/lib"
[ -e "$cudalib/libcudart.so.13" ] && ln -sf libcudart.so.13 "$cudalib/libcudart.so"
