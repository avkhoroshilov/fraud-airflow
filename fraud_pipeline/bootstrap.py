"""Load the supplied labeled CSV in chunks and initialize a cold-start model."""
import hashlib
import logging
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from fraud_pipeline.features import REQUIRED, CATEGORICAL, build_model
from fraud_pipeline.storage import (connect, get_state, insert_frame, month_add, read_frame,
                                    root, save_model, schema, set_state, digest)

logger = logging.getLogger(__name__)


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def initialize():
    """Read train.csv; atomically load all rows, source metadata and baseline.

    CSV is read in 10000-row chunks. Only required model inputs and target are
    stored. The transaction commits after the baseline file is durable. A crash
    rolls the load back; an initialized CSV volume is never loaded twice.
    """
    schema()
    path = Path(os.getenv("TRAIN_DATA_CSV", "/input/train.csv"))
    with connect(write=True) as c:
        if get_state(c, "initialized"):
            source = get_state(c, "data_source")
            if not source or source.get("kind") != "csv":
                raise RuntimeError("Existing volume contains demo data. Start the new fraud-airflow-csv Compose project.")
            logger.info("CSV mart already initialized: %s rows", source["rows"])
            return
        if not path.is_file():
            raise FileNotFoundError("Put the labeled train.csv in the project's data/ directory; expected "+str(path))
        header = pd.read_csv(path, nrows=0).columns
        required = set(REQUIRED + ["target"])
        missing = required-set(header)
        if missing:
            raise ValueError(f"train.csv lacks required fields: {sorted(missing)}")
        window = int(os.getenv("WINDOW_MONTHS", "3"))
        history = int(os.getenv("HISTORY_MONTHS", "36"))
        if window < 2 or history < window:
            raise ValueError("Require HISTORY_MONTHS >= WINDOW_MONTHS >= 2")
        columns = required | ({"transaction_id"} if "transaction_id" in header else set())
        types = {name: str for name in CATEGORICAL}
        if "transaction_id" in columns:
            types["transaction_id"] = str
        total = 0
        months = Counter()
        labels = Counter()
        for frame in pd.read_csv(path, usecols=lambda name: name in columns, dtype=types, chunksize=10000):
            times = pd.to_datetime(frame.transaction_time, errors="raise", utc=True)
            if times.isna().any():
                raise ValueError("train.csv has missing transaction_time")
            frame["transaction_time"] = times.dt.strftime("%Y-%m-%d %H:%M:%S")
            target = pd.to_numeric(frame.target, errors="raise")
            if not target.isin([0,1]).all():
                raise ValueError("train.csv target must contain only 0/1")
            frame["target"] = target.astype(int)
            if "transaction_id" not in frame:
                frame["transaction_id"] = [f"csv-{total+i}" for i in range(len(frame))]
            elif frame.transaction_id.isna().any() or frame.transaction_id.str.strip().eq("").any():
                raise ValueError("train.csv has empty transaction_id")
            insert_frame(c, frame)
            months.update(frame.transaction_time.str[:7])
            labels.update(frame.target)
            total += len(frame)
        available = sorted(months)
        first = available[:window]
        if len(first) < window or first != [str(p) for p in pd.period_range(first[0], periods=window, freq="M")]:
            raise ValueError("train.csv needs consecutive months for the initial window")
        if history < len(available):
            raise ValueError("CSV exceeds HISTORY_MONTHS; trim it or increase retention")
        train = read_frame(c, first[0]+"-01", first[-1]+"-01")
        if set(train.target.unique()) != {0,1}:
            raise ValueError("Initial training data needs both target classes")
        model = build_model().set_params(classifier=DummyClassifier(strategy="prior")).fit(train, train.target)
        model_path = root()/"models/model_initial.joblib"
        save_model(model_path, model)
        source = dict(kind="csv", filename=path.name, rows=total, sha256=file_hash(path),
                      months=dict(sorted(months.items())), labels={str(k):v for k,v in sorted(labels.items())})
        set_state(c,"active",dict(version="initial",path=str(model_path),sha256=digest(model_path)))
        set_state(c,"next_month",month_add(first[-1],1))
        set_state(c,"window_months",window)
        set_state(c,"history_months",history)
        set_state(c,"data_source",source)
        set_state(c,"initialized",True)
    logger.info("Loaded %s labeled CSV rows; initial window %s", total, first)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize()
