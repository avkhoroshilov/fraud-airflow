"""Independent scheduled source of the next labeled calendar month."""
from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.decorators import task
from fraud_pipeline.tasks.supply import prepare_supply, commit_supply

with DAG("supply_labeled_batch", schedule="*/5 * * * *", start_date=datetime(2024,1,1,tzinfo=timezone.utc),
         catchup=False, max_active_runs=1,
         default_args={"retries": 2, "retry_delay": timedelta(seconds=20)}, tags=["fraud", "data"]) as dag:
    batch = task(prepare_supply)(run_id="{{ run_id }}")
    task(commit_supply)(batch)
