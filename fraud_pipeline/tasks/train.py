"""Training never sees the evaluation month."""
from pathlib import Path
import pandas as pd
from fraud_pipeline.features import build_model
from fraud_pipeline.storage import save_model


def train_candidate(info):
    """Read train.json; fit preprocessing and classifier; save a candidate binary.

    The held-out month is not passed to fit or hyperparameter selection. Repeats
    overwrite the same private candidate; they cannot publish or advance state.
    """
    directory = Path(info["directory"])
    train = pd.read_json(directory / "train.json", convert_dates=False, dtype=False)
    model = build_model().fit(train, train.target)
    save_model(directory / "candidate.joblib", model)
    return info
