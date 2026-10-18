"""Send one uniquely identified event and verify its committed model version."""
import argparse
import csv
import json
import os
import time
import uuid
from pathlib import Path
from confluent_kafka import Consumer, Producer
from fraud_pipeline.storage import connect, get_state, sample_frame
from fraud_pipeline.stream import send, BROKERS

parser = argparse.ArgumentParser()
parser.add_argument("--require-trained", action="store_true")
parser.add_argument("--csv", default=os.getenv("TRANSACTIONS_CSV", "/input/test.csv"))
args = parser.parse_args()
with connect() as c:
    expected = get_state(c,"active")["version"]
    if Path(args.csv).is_file():
        with Path(args.csv).open(encoding="utf-8-sig", newline="") as source:
            row = next(csv.DictReader(source))
    else:
        row = sample_frame(c,1,"verify").iloc[0].to_dict()
if args.require_trained and expected == "initial":
    raise RuntimeError("Expected a published feature-based model, found initial")
row.pop("target",None)
row.pop("transaction_id",None)
ident = "verify-"+str(uuid.uuid4())
consumer = Consumer({"bootstrap.servers":BROKERS,"group.id":ident,
                     "auto.offset.reset":"earliest","enable.auto.commit":False})
consumer.subscribe(["scoring"])
send(Producer({"bootstrap.servers":BROKERS}),"transactions",dict(transaction_id=ident,data=row),ident)
deadline = time.monotonic()+90
try:
    while time.monotonic()<deadline:
        message=consumer.poll(1)
        if message is None:
            continue
        if message.error():
            raise RuntimeError(message.error())
        for result in json.loads(message.value()):
            if result["transaction_id"]==ident:
                assert result["model_version"]==expected,result
                assert 0<=result["score"]<=1,result
                assert result["fraud_flag"] in (0,1),result
                print(json.dumps(result,ensure_ascii=False))
                raise SystemExit(0)
    raise RuntimeError("Timed out waiting for the scored verification event")
finally:
    consumer.close()
