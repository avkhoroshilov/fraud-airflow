"""Publication and training progress share one SQLite commit boundary."""
import json
from pathlib import Path
from fraud_pipeline.storage import (atomic_bytes, connect, digest, get_state,
                                    month_add, root, set_state, write_json)


def publish_candidate(info):
    """Read gate + staged binary; persist model, then atomically commit pointers.

    Rejection leaves production and month unchanged. Durable files precede the
    DB commit; a crash leaves at most an unreferenced file. An accepted run's
    retry is a no-op, including after later versions have been published.
    """
    directory = Path(info["directory"])
    decision = json.loads((directory / "metrics.json").read_text())
    with connect(write=True) as c:
        row = c.execute("SELECT * FROM training_runs WHERE run_id=?", (info["run_id"],)).fetchone()
        if row is None or row["version"] != info["version"] or row["month"] != info["month"]:
            raise RuntimeError("Unknown or mismatching training reservation")
        if row["status"] in ("published", "rejected"):
            return dict(version=info["version"], status=row["status"])
        if get_state(c, "next_month") != info["month"] or get_state(c, "active")["version"] != info["champion"]:
            raise RuntimeError("Stale candidate: current month/champion has changed")
        if not decision["accepted"]:
            c.execute("UPDATE training_runs SET status='rejected' WHERE run_id=?", (info["run_id"],))
            return dict(version=info["champion"], status="rejected")
        candidate = directory / "candidate.joblib"
        if digest(candidate) != decision["candidate_sha256"]:
            raise RuntimeError("Candidate changed after evaluation")
        dest = root() / "models" / f"model_{info['version']}.joblib"
        if dest.exists():
            if digest(dest) != decision["candidate_sha256"]:
                raise RuntimeError("Immutable model file mismatch")
        else:
            atomic_bytes(dest, lambda f: f.write(candidate.read_bytes()))
        write_json(dest.with_suffix(".metrics.json"), decision)
        set_state(c, "active", dict(version=info["version"], path=str(dest), sha256=digest(dest)))
        # This and active are in the SAME transaction; model already fsynced.
        set_state(c, "next_month", month_add(info["month"], 1))
        c.execute("UPDATE training_runs SET status='published' WHERE run_id=?", (info["run_id"],))
    return dict(version=info["version"], status="published")
