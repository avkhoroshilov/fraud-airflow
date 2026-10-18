"""Month reservation and an atomic append plus retention transaction."""
import os
import pandas as pd
from fraud_pipeline.storage import (connect, get_state, insert_frame, month_add, sample_frame,
                                    run_dir, skip, write_json)


def prepare_supply(run_id):
    """Read existing labeled rows; stage a deterministic next calendar month.

    A durable run reservation fixes its month across retries. Committed runs
    return their old month; a retry cannot append a second month.
    """
    directory = run_dir(run_id, "supply")
    with connect(write=True) as c:
        record = c.execute("SELECT * FROM supply_runs WHERE run_id=?", (run_id,)).fetchone()
        if record:
            month = record["month"]
            path = directory / f"batch_{month}.json"
            if record["status"] == "committed":
                return {"run_id": run_id, "month": month, "path": str(path)}
        else:
            pending = c.execute("SELECT run_id FROM supply_runs WHERE status='pending'").fetchone()
            if pending:
                raise RuntimeError(f"Finish/clear failed supply run first: {pending[0]}")
            latest = c.execute("SELECT substr(max(transaction_time),1,7) FROM labeled_transactions").fetchone()[0]
            month = month_add(latest, 1)
            cutoff = month_add(month, 1-get_state(c,"history_months"))
            needed = month_add(get_state(c,"next_month"), 1-get_state(c,"window_months"))
            if cutoff > needed:
                skip("Waiting for retraining: retention would remove its pending window")
            c.execute("INSERT INTO supply_runs VALUES (?,?,'pending')", (run_id, month))
        path = directory / f"batch_{month}.json"
        if not path.exists():
            batch = sample_frame(c, int(os.getenv("BATCH_SIZE", "2000")), month).reset_index(drop=True)
            n = len(batch)
            if not n:
                raise RuntimeError("Labeled mart is empty")
            t = pd.to_datetime(batch.transaction_time)
            target = pd.Period(month, freq="M")
            batch["transaction_time"] = pd.to_datetime(dict(
                year=[target.year]*n, month=[target.month]*n,
                day=t.dt.day.clip(upper=target.days_in_month), hour=t.dt.hour,
                minute=t.dt.minute, second=t.dt.second)).dt.strftime("%Y-%m-%d %H:%M:%S")
            batch["transaction_id"] = [f"supply-{month}-{i}" for i in range(n)]
            write_json(path, batch.to_dict("records"))
    return {"run_id": run_id, "month": month, "path": str(path)}


def commit_supply(info):
    """Read staged rows; append, prune and mark the run in one transaction.

    Already committed run IDs are no-ops. No partial batch or repeated pruning
    survives rollback; the training window is protected by supply backpressure.
    """
    with connect(write=True) as c:
        row = c.execute("SELECT * FROM supply_runs WHERE run_id=?", (info["run_id"],)).fetchone()
        if not row or row["month"] != info["month"]:
            raise RuntimeError("Unknown or mismatching supply reservation")
        if row["status"] == "committed":
            return info
        frame = pd.read_json(info["path"], convert_dates=False, dtype=False)
        if frame.empty or set(frame.transaction_time.str[:7]) != {info["month"]}:
            raise ValueError("Staged batch has wrong month or no rows")
        insert_frame(c, frame)
        cutoff = month_add(info["month"], 1-get_state(c, "history_months")) + "-01"
        c.execute("DELETE FROM labeled_transactions WHERE transaction_time<?", (cutoff,))
        c.execute("UPDATE supply_runs SET status='committed' WHERE run_id=?", (info["run_id"],))
    return info
