"""Local-socket request/response helper for long-lived inference worker
subprocesses that render_objaverse.py spawns (see clip_worker.py /
aesthetic_worker.py). Stdlib-only so it also imports fine from Blender's
bundled Python.

Each worker: binds a Unix domain socket, prints "READY" to stdout once
bound (the parent waits for this line rather than polling for the socket
file to appear -- render_objaverse.py used to do exactly that via rpyc, see
the module docstrings this replaces), accepts exactly one connection, then
loops recv()/send() until the parent closes its end (EOF), at which point
the worker exits cleanly.

Small messages (embeddings, scores, control pings) go over the socket via
multiprocessing.connection, which pickles whatever picklable Python object
you pass -- no hand-rolled framing, and no RPC-style proxying (that's what
bit this project's old rpyc-based render_server.py: returning a nested list
over rpyc silently became one round trip per element). The one payload
large enough to matter -- a rendered RGBA frame, up to a few MB -- bypasses
the socket entirely via shared_frame_buffer() instead.

Numpy arrays are never sent as objects over the socket: Blender's bundled
Python and the project's venv Python have separate numpy installs, and
pickled-ndarray compatibility across them isn't guaranteed. Everything here
moves as plain bytes/tuples/floats/None, interpreted independently as numpy
arrays on each side via .tobytes()/np.frombuffer()/a shared buffer view.
"""

import contextlib as ctl
import itertools
import logging
import os
import time

log = logging.getLogger("ipc")

_shm_counter = itertools.count()


class WorkerDiedError(RuntimeError):
  """The worker subprocess is gone (failed to start, or died mid-run) --
  distinct from a bad request, which is recoverable."""


def serve_forever(socket_path, handle):
  """Worker side. Binds `socket_path`, prints "READY" once listening,
  accepts exactly one connection, then calls `handle(request)` for every
  message received until the client disconnects (EOFError), at which point
  the socket is closed and this returns."""
  from multiprocessing.connection import Listener

  listener = Listener(socket_path, family="AF_UNIX")
  try:
    print("READY", flush=True)
    conn = listener.accept()
    try:
      while True:
        try:
          request = conn.recv()
        except EOFError:
          break
        conn.send(handle(request))
    finally:
      conn.close()
  finally:
    listener.close()


@ctl.contextmanager
def worker_connection(python, script, socket_path, *args, ready_timeout=300.0):
  """Client side (render_objaverse.py). Spawns
  `python script socket_path *args`, waits for its "READY" line on stdout
  (worker stderr is left inherited, so a worker traceback shows up directly
  in the console -- Python's logging defaults to stderr, so it never
  pollutes the READY-line channel), connects a Client to `socket_path`, and
  yields the Connection. `ready_timeout` is generous (300s default) because
  a worker's first run may need to download model weights, not just load
  them.

  On exit: closes the connection (signals EOF, so a still-healthy worker
  exits its own serve_forever loop on its own), then waits up to 10s for
  the subprocess to exit before killing it.
  """
  import select
  import subprocess
  from multiprocessing.connection import Client

  proc = subprocess.Popen(
    [python, script, socket_path, *args], stdout=subprocess.PIPE, text=True)
  try:
    deadline = time.monotonic() + ready_timeout
    while True:
      if proc.poll() is not None:
        raise WorkerDiedError(
          f"{script} exited (code {proc.returncode}) before printing READY")
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        proc.kill()
        raise WorkerDiedError(f"{script} did not print READY within {ready_timeout}s")
      # select() bounds the wait so the poll()/deadline checks above actually
      # re-run periodically -- a plain blocking readline() wouldn't notice
      # either a dead process or an expired deadline until a line arrived.
      readable, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
      if not readable:
        continue
      line = proc.stdout.readline()
      if line.strip() == "READY":
        break

    conn = Client(socket_path, family="AF_UNIX")
    try:
      yield conn
    finally:
      conn.close()
  finally:
    if proc.poll() is None:
      try:
        proc.wait(timeout=10)
      except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@ctl.contextmanager
def shared_frame_buffer(nbytes):
  """Allocate a POSIX shared-memory block big enough for one RGBA frame,
  named distinctively ("gsviawt_clip_<pid>_<n>") so a segment leaked by a
  hard kill/crash (which skips unlink) is easy to spot and clean up by hand
  under /dev/shm. The caller writes each frame in with a persistent
  np.ndarray(shape, uint8, buffer=shm.buf) view (created once, reused every
  call -- no per-frame allocation); clip_worker.py attaches by `shm.name`
  and reads the same bytes zero-copy.

  Must be entered *before*, and therefore (stack.enter_context LIFO order)
  closed/unlinked *after*, the worker_connection it feeds -- unlinking
  while the worker still has it open is a use-after-free. Safety of the
  single shared slot relies on the protocol being strictly synchronous: the
  caller must never overwrite the buffer until it has already received the
  worker's response for the previous frame (no pipelining, no concurrent
  read/write)."""
  from multiprocessing.shared_memory import SharedMemory

  name = f"gsviawt_clip_{os.getpid()}_{next(_shm_counter)}"
  shm = SharedMemory(name=name, create=True, size=nbytes)
  try:
    yield shm
  finally:
    shm.close()
    try:
      shm.unlink()
    except FileNotFoundError:
      pass
