"""Hot reload uses the same committed registry row as the training pointer."""
import logging
import threading
import joblib
import pandas as pd
from fraud_pipeline.storage import connect, get_state, digest

logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self):
        self.model = None
        self.version = None
        self.lock = threading.Lock()

    def score(self, message):
        with self.lock:
            with connect() as c:
                active = get_state(c, "active")
            if active is None:
                raise RuntimeError("No published model; run data-init")
            if active["version"] != self.version:
                try:
                    if digest(active["path"]) != active["sha256"]:
                        raise ValueError("Model checksum mismatch")
                    new = joblib.load(active["path"])
                    # Assign model and version together only after successful loading.
                    self.model, self.version = new, active["version"]
                    logger.info("Loaded model_version=%s", self.version)
                except Exception as exc:
                    if self.model is None:
                        raise RuntimeError("Cannot load the active model at startup") from exc
                    logger.exception("Reload failed; retaining model_version=%s", self.version)
            probability = float(self.model.predict_proba(pd.DataFrame([message["data"]]))[0,1])
            return dict(transaction_id=message["transaction_id"], score=probability,
                        fraud_flag=int(probability >= .5), model_version=self.version)
