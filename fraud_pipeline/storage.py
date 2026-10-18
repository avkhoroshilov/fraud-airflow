"""One local shared volume, SQLite transactions and immutable model files."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import joblib
import pandas as pd


def root():
    return Path(os.getenv("SHARED_DIR", "/shared"))


@contextlib.contextmanager
def connect(write=False):
    root().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root() / "state.sqlite", timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def schema():
    with connect() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript('''
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS labeled_transactions(
          transaction_id TEXT PRIMARY KEY, transaction_time TEXT NOT NULL,
          target INTEGER NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS labeled_time ON labeled_transactions(transaction_time);
        CREATE TABLE IF NOT EXISTS supply_runs(
          run_id TEXT PRIMARY KEY, month TEXT UNIQUE NOT NULL, status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS training_runs(
          run_id TEXT PRIMARY KEY, month TEXT NOT NULL, champion TEXT,
          status TEXT NOT NULL, version TEXT NOT NULL);
        ''')


def get_state(c, key, default=None):
    row = c.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_state(c, key, value):
    c.execute("INSERT INTO state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
              (key, json.dumps(value, allow_nan=False)))


def month_add(month, n):
    return str(pd.Period(month, freq="M") + n)


def read_frame(c, start=None, end=None):
    rows = c.execute(
        "SELECT transaction_id,payload FROM labeled_transactions "
        "WHERE (? IS NULL OR transaction_time>=?) AND (? IS NULL OR transaction_time<?) "
        "ORDER BY transaction_time,transaction_id", (start, start, end, end)).fetchall()
    return pd.DataFrame([dict(json.loads(r["payload"]), transaction_id=r["transaction_id"]) for r in rows])


def iter_frames(c, size=5000):
    """Read the labeled mart in bounded chunks within the caller's transaction."""
    cursor = c.execute("SELECT transaction_id,payload FROM labeled_transactions ORDER BY transaction_time,transaction_id")
    while True:
        rows = cursor.fetchmany(size)
        if not rows:
            return
        yield pd.DataFrame([dict(json.loads(r["payload"]), transaction_id=r["transaction_id"]) for r in rows])


def sample_frame(c, size, seed):
    """Select a reproducible sample without materializing the whole mart."""
    if size < 1:
        raise ValueError("Sample size must be positive")
    c.create_function("sample_key", 1, lambda ident: hashlib.sha256((str(seed)+ident).encode()).digest())
    rows = c.execute("SELECT transaction_id,payload FROM labeled_transactions "
                     "ORDER BY sample_key(transaction_id) LIMIT ?", (size,)).fetchall()
    return pd.DataFrame([dict(json.loads(r["payload"]), transaction_id=r["transaction_id"]) for r in rows])


def insert_frame(c, frame):
    rows = []
    for record in frame.to_dict("records"):
        record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        ident = record.pop("transaction_id")
        rows.append((ident, record["transaction_time"], int(record["target"]),
                     json.dumps(record, allow_nan=False)))
    c.executemany("INSERT INTO labeled_transactions VALUES (?,?,?,?)", rows)


def atomic_bytes(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic_bytes(path, lambda f: f.write(json.dumps(value, ensure_ascii=False, indent=2,
                                                  allow_nan=False).encode()))


def save_model(path, model):
    atomic_bytes(path, lambda f: joblib.dump(model, f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_dir(run_id, kind="training"):
    token = hashlib.sha256(run_id.encode()).hexdigest()[:20]
    path = root() / "work" / kind / token
    path.mkdir(parents=True, exist_ok=True)
    return path


def skip(message):
    # Airflow is optional for local unit/integration checks.
    try:
        from airflow.exceptions import AirflowSkipException
    except ImportError:
        raise RuntimeError(message)
    raise AirflowSkipException(message)
