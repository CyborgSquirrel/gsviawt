#!/usr/bin/env python3
"""Long-lived CLIP ViT-L/14 (OpenAI weights) embedding server used by
render_objaverse.py's compute_clip. Spawned once per render run (via
ipc.worker_connection, using the project's normal venv python -- not
Blender's bundled one, see render_objaverse.py's docstring), loads the
model once, then scores one rendered view per request for the lifetime of
the render.

Not meant to be run by hand -- render_objaverse.py spawns it with:

    python clip_worker.py <socket_path> <shm_name> <width> <height> \\
        <bg_r> <bg_g> <bg_b>

`shm_name` names a multiprocessing.shared_memory block (allocated by the
parent, sized width*height*4 bytes) holding the current RGBA frame -- see
ipc.shared_frame_buffer. Protocol: binds a Unix socket at `socket_path` via
ipc.serve_forever, prints "READY" once listening and the frame buffer is
attached, then loops:
  recv() -> None (a ready ping; the frame is already sitting in shared memory)
  send() -> bytes (768 float32s, little-endian, the CLIP embedding)
Connection close (EOF) ends the loop and the process exits 0.
"""

import sys

import numpy as np

import ipc
from clip_score import composite, embed_batch, load_clip_model, resolve_device


def main():
  socket_path, shm_name, width, height, bg_r, bg_g, bg_b = sys.argv[1:8]
  width, height = int(width), int(height)
  background = [float(bg_r), float(bg_g), float(bg_b)]

  device = resolve_device("auto")
  model, preprocess = load_clip_model(device)

  from multiprocessing import resource_tracker
  from multiprocessing.shared_memory import SharedMemory
  shm = SharedMemory(name=shm_name, create=False)
  # SharedMemory registers with this process's resource_tracker on attach
  # too, not just on create -- and only unlink() (which only the owning
  # parent calls) unregisters it. Left as-is, this process's tracker would
  # warn "leaked shared_memory" at exit even though the parent cleans the
  # segment up correctly; unregister here since we're an attacher, not the
  # owner (see https://bugs.python.org/issue38119).
  resource_tracker.unregister(shm._name, "shared_memory")
  # Zero-copy read view onto the frame the parent writes each request --
  # composite() builds a new float32 array rather than mutating its input,
  # so reading straight out of shared memory here is safe.
  frame = np.ndarray((height, width, 4), dtype=np.uint8, buffer=shm.buf)

  def handle(_ping):
    rgb = composite(frame[None], background)
    emb = embed_batch(model, preprocess, rgb, device)[0]
    return emb.astype("<f4").tobytes()

  try:
    ipc.serve_forever(socket_path, handle)
  finally:
    shm.close()


if __name__ == "__main__":
  main()
