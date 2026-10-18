"""Compare challenger and champion on precisely the same held-out month."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from fraud_pipeline.storage import digest, root, write_json


def metrics(model, frame):
    proba = model.predict_proba(frame)[:, 1]
    return {"roc_auc": float(roc_auc_score(frame.target, proba)),
            "average_precision": float(average_precision_score(frame.target, proba)),
            "f1": float(f1_score(frame.target, proba >= .5, zero_division=0))}


def passes_gate(candidate, champion):
    # Zero tolerance for measured regression on either ranking metric.
    return all(candidate[k] >= champion[k] for k in ("roc_auc", "average_precision"))


def evaluate_candidate(info):
    """Read candidate/champion and test.json; write a reproducible gate decision.

    Both models use the identical fresh month. Equality is allowed. Repeating
    evaluation rewrites only this run's metrics and cannot change production.
    """
    directory = Path(info["directory"])
    test = pd.read_json(directory / "test.json", convert_dates=False, dtype=False)
    candidate = metrics(joblib.load(directory / "candidate.joblib"), test)
    champion_path = root() / "models" / f"model_{info['champion']}.joblib"
    champion = metrics(joblib.load(champion_path), test)
    decision = dict(candidate=candidate, champion=champion, accepted=passes_gate(candidate, champion),
                    test_month=info["month"], test_rows=len(test),
                    train_months=[str(m) for m in pd.period_range(info["start"], periods=info["months"]-1, freq="M")],
                    candidate_sha256=digest(directory / "candidate.joblib"))
    write_json(directory / "metrics.json", decision)
    return info
