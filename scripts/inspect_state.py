"""Display model pointer, next month and per-month labeled row counts."""
import json
from fraud_pipeline.storage import connect, get_state

with connect() as c:
    print(json.dumps({"active": get_state(c,"active"), "next_month": get_state(c,"next_month"),
                      "data_source": get_state(c,"data_source"),
                      "months": [dict(r) for r in c.execute(
                          "SELECT substr(transaction_time,1,7) month,count(*) rows "
                          "FROM labeled_transactions GROUP BY month ORDER BY month")],
                      "training_runs": [dict(r) for r in c.execute("SELECT * FROM training_runs")]}, indent=2))
