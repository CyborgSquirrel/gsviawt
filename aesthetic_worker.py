#!/usr/bin/env python3
"""Long-lived LAION improved-aesthetic-predictor MLP server used by
render_objaverse.py's compute_aesthetic. Spawned once per render run
alongside (but independently of) clip_worker.py -- it does NOT load CLIP
itself, only the small MLP head, and scores embeddings that
render_objaverse.py already got back from clip_worker.py.

Not meant to be run by hand -- render_objaverse.py spawns it with:

    python aesthetic_worker.py <socket_path>

Protocol: binds a Unix socket at `socket_path` via ipc.serve_forever, prints
"READY" once the MLP weights are loaded, then loops:
  recv() -> bytes (768 float32s, little-endian -- a CLIP ViT-L/14 embedding)
  send() -> float (the predicted aesthetic score)
Connection close (EOF) ends the loop and the process exits 0.
"""

import sys

import numpy as np

import ipc
from clip_score import aesthetic_scores, load_aesthetic_mlp, resolve_device


def main():
  socket_path, = sys.argv[1:2]

  device = resolve_device("auto")
  mlp = load_aesthetic_mlp(device)

  def handle(emb_bytes):
    # np.frombuffer views the (immutable) bytes object directly, giving a
    # read-only array -- copy so torch.from_numpy (inside aesthetic_scores)
    # doesn't warn about writing to it.
    emb = np.frombuffer(emb_bytes, dtype="<f4").reshape(1, -1).copy()
    return float(aesthetic_scores(mlp, emb)[0])

  ipc.serve_forever(socket_path, handle)


if __name__ == "__main__":
  main()
