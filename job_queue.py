"""A small job queue on a SQLite file, safe across processes (workers claim jobs concurrently).

    q = JobQueue("/app/bla/queue.db")
    q.add([(priority, {"mesh_index": 0, "path": "a.glb"}), ...])   # higher priority is claimed first
    job = q.claim("worker-7")       # {"id": ..., "payload": {...}} or None when nothing is ready
    q.done(job, "worker-7")         # or q.fail(job, "worker-7", "why")
    q.requeue_stale({"worker-3"})   # claimed by any other worker -> ready again
    q.counts()                      # {"ready": .., "claimed": .., "done": .., "failed": ..}

A claim is one atomic UPDATE ... RETURNING, so two processes never get the same job. Nothing is
resumed implicitly: a job claimed by a worker that died stays "claimed" (with its worker and claim
time) until requeue_stale() is called with the workers that are still alive. done()/fail() only
take effect for the worker that holds the claim, so a worker that was presumed dead can't undo a
requeue. A job is whatever JSON payload the caller wants; `meta` holds shared JSON values (e.g. the
full mesh list the jobs index into).
"""

import json
import sqlite3
import time


class JobQueue:
  STATUSES = ("ready", "claimed", "done", "failed")

  def __init__(self, path):
    self.path = path
    self.db = sqlite3.connect(path, timeout=60, isolation_level=None)   # autocommit: one statement, one transaction
    self.db.execute("PRAGMA journal_mode=WAL")
    self.db.execute("PRAGMA busy_timeout=60000")
    self.db.execute(
      "CREATE TABLE IF NOT EXISTS jobs ("
      " id INTEGER PRIMARY KEY, priority REAL NOT NULL, payload TEXT NOT NULL,"
      " status TEXT NOT NULL DEFAULT 'ready', worker TEXT, claimed_at REAL, error TEXT)")
    self.db.execute("CREATE INDEX IF NOT EXISTS jobs_ready ON jobs (status, priority DESC, id)")
    self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

  def close(self):
    self.db.close()

  def set_meta(self, key, value):
    self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value)))

  def get_meta(self, key):
    row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
      raise KeyError(key)
    return json.loads(row[0])

  def add(self, jobs):
    """jobs: iterable of (priority, payload); all-or-nothing."""
    rows = [(float(priority), json.dumps(payload)) for priority, payload in jobs]
    self.db.execute("BEGIN IMMEDIATE")
    try:
      self.db.executemany("INSERT INTO jobs (priority, payload) VALUES (?, ?)", rows)
    except BaseException:
      self.db.execute("ROLLBACK")
      raise
    self.db.execute("COMMIT")

  def claim(self, worker):
    row = self.db.execute(
      "UPDATE jobs SET status = 'claimed', worker = ?, claimed_at = ?"
      " WHERE id = (SELECT id FROM jobs WHERE status = 'ready' ORDER BY priority DESC, id LIMIT 1)"
      " RETURNING id, payload", (str(worker), time.time())).fetchone()
    if row is None:
      return None
    return {"id": row[0], "payload": json.loads(row[1])}

  def _finish(self, status, job, worker, error=None):
    cur = self.db.execute(
      "UPDATE jobs SET status = ?, error = ? WHERE id = ? AND status = 'claimed' AND worker = ?",
      (status, error, job["id"], str(worker)))
    return cur.rowcount == 1

  def done(self, job, worker):
    """True if `worker` still held the claim (the job is now done), False if it was requeued."""
    return self._finish("done", job, worker)

  def fail(self, job, worker, error):
    """Mark a job as failed for good (no retry); same return value as done()."""
    return self._finish("failed", job, worker, str(error))

  def requeue_stale(self, alive_workers):
    """Claimed by a worker not in alive_workers -> ready again. Returns the number of jobs requeued."""
    alive = [str(w) for w in alive_workers]
    cond = f" AND worker NOT IN ({','.join('?' * len(alive))})" if alive else ""
    cur = self.db.execute(
      "UPDATE jobs SET status = 'ready', worker = NULL, claimed_at = NULL WHERE status = 'claimed'" + cond, alive)
    return cur.rowcount

  def claims(self):
    """(job id, worker, claimed_at) of every claimed job -- to spot jobs held for suspiciously long."""
    return self.db.execute("SELECT id, worker, claimed_at FROM jobs WHERE status = 'claimed'").fetchall()

  def failed(self):
    """(job id, payload, error) of every failed job."""
    return [(i, json.loads(p), e) for i, p, e in
            self.db.execute("SELECT id, payload, error FROM jobs WHERE status = 'failed' ORDER BY id")]

  def counts(self):
    got = dict(self.db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
    return {s: got.get(s, 0) for s in self.STATUSES}
