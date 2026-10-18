"""Compatible seminar Kafka envelopes, acknowledged output and manual offsets."""
import csv
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from confluent_kafka import Consumer, Producer
from fraud_pipeline.scoring import Scorer
from fraud_pipeline.storage import connect, sample_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BROKERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")


def send(producer, topic, value, key):
    errors = []
    def delivered(error, message):
        if error:
            errors.append(str(error))
    producer.produce(topic, key=str(key), value=json.dumps(value, allow_nan=False), callback=delivered)
    if producer.flush(30) or errors:
        raise RuntimeError(f"Kafka delivery failed: {errors}")


def csv_sample(path, size=2000):
    """Reservoir sample of input CSV rows; labels are never sent to inference."""
    rng = random.Random(42)
    records = []
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for i, row in enumerate(csv.DictReader(source)):
            row.pop("target", None)
            row.pop("transaction_id", None)
            if len(records) < size:
                records.append(row)
            else:
                position = rng.randrange(i+1)
                if position < size:
                    records[position] = row
    if not records:
        raise ValueError("Transaction CSV is empty")
    return records


def simulator():
    path = Path(os.getenv("TRANSACTIONS_CSV", "/input/test.csv"))
    if path.is_file():
        records = csv_sample(path)
        logger.info("Streaming transactions from %s", path)
    else:
        with connect() as c:
            records = sample_frame(c, 2000, "simulator").drop(columns=["target", "transaction_id"]).to_dict("records")
        logger.info("No test.csv; streaming a sample from the labeled mart")
    producer = Producer({"bootstrap.servers": BROKERS, "enable.idempotence": True})
    while True:
        row = dict(random.choice(records))
        row["transaction_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        ident = str(uuid.uuid4())
        send(producer, "transactions", {"transaction_id": ident, "data": row}, ident)
        time.sleep(float(os.getenv("SEND_INTERVAL_SECONDS", ".5")))


def scorer_loop():
    consumer = Consumer({"bootstrap.servers": BROKERS, "group.id": "fraud-scorer-v1",
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    producer = Producer({"bootstrap.servers": BROKERS, "enable.idempotence": True})
    scorer = Scorer()
    consumer.subscribe(["transactions"])
    try:
        while True:
            message = consumer.poll(1)
            if message is None:
                continue
            if message.error():
                raise RuntimeError(message.error())
            try:
                incoming = json.loads(message.value())
                result = scorer.score(incoming)
            except (ValueError, KeyError, TypeError) as exc:
                # Malformed input is reviewable and does not permanently block a partition.
                send(producer, "scoring_errors", {"error": str(exc), "input": message.value().decode(errors="replace")},
                     f"{message.partition()}:{message.offset()}")
            else:
                # Seminar output contract: JSON array of results.
                send(producer, "scoring", [result], result["transaction_id"])
                logger.info("Scored %s model_version=%s", result["transaction_id"], result["model_version"])
            consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()


if __name__ == "__main__":
    {"simulator": simulator, "scorer": scorer_loop}[sys.argv[1]]()
