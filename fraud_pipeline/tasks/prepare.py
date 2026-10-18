"""Freeze an immutable, run-specific window and chronological split."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from fraud_pipeline.storage import (connect, get_state, month_add, read_frame,
                                    run_dir, skip, write_json)


def prepare_window(run_id):
    """Read next_month and labeled rows; snapshot exactly WINDOW_MONTHS.

    The reservation fixes month/champion per run. No training pointer changes.
    Run-specific files and retained snapshots make clearing tasks safe.
    """
    base_directory = run_dir(run_id)
    with connect(write=True) as c:
        row = c.execute("SELECT * FROM training_runs WHERE run_id=?", (run_id,)).fetchone()
        if row:
            month, champion, version = row["month"], row["champion"], row["version"]
        else:
            pending = c.execute("SELECT run_id FROM training_runs WHERE status='pending'").fetchone()
            if pending:
                raise RuntimeError(f"Finish/clear failed training run first: {pending[0]}")
            month = get_state(c, "next_month")
            champion = get_state(c, "active")["version"]
            version = month + "-" + hashlib.sha256(run_id.encode()).hexdigest()[:12]
        count = get_state(c, "window_months")
        start = month_add(month, 1-count)
        directory = base_directory / month
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory / "window.json"
        if not snapshot.exists():
            frame = read_frame(c, start+"-01", month_add(month, 1)+"-01")
            expected = [month_add(start, i) for i in range(count)]
            if frame.empty or sorted(frame.transaction_time.str[:7].unique()) != expected:
                skip(f"Window {expected} is incomplete; supply next month first")
            write_json(snapshot, frame.to_dict("records"))
        if not row:
            c.execute("INSERT INTO training_runs VALUES (?,?,?,'pending',?)",
                      (run_id, month, champion, version))
    return dict(run_id=run_id, month=month, champion=champion, version=version,
                directory=str(directory), start=start, months=count)


def split_by_time(info):
    """Read the frozen window; write earlier months to train and latest to test.

    Rewrites only this run's deterministic files. It never changes registry state.
    """
    directory = Path(info["directory"])
    frame = pd.read_json(directory / "window.json", convert_dates=False, dtype=False)
    months = frame.transaction_time.str[:7]
    train, test = frame.loc[months < info["month"]], frame.loc[months == info["month"]]
    if train.empty or test.empty or train.transaction_time.max() >= test.transaction_time.min():
        raise ValueError("Invalid chronological split")
    if set(train.target.unique()) != {0, 1} or set(test.target.unique()) != {0, 1}:
        raise ValueError("Both train and test need both target classes")
    write_json(directory / "train.json", train.to_dict("records"))
    write_json(directory / "test.json", test.to_dict("records"))
    return info
