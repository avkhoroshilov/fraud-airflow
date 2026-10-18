"""HTML publication precedes failure enforcement, including on bad data."""
from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.decorators import task
from fraud_pipeline.tasks.quality import snapshot_quality, evaluate_quality, render_quality, enforce_quality

with DAG("fraud_data_quality", schedule="2-59/5 * * * *", start_date=datetime(2024,1,1,tzinfo=timezone.utc),
         catchup=False, max_active_runs=1,
         default_args={"retries": 1, "retry_delay": timedelta(seconds=20)}, tags=["fraud", "dq"]) as dag:
    snapshot = task(snapshot_quality)(run_id="{{ run_id }}")
    checked = task(evaluate_quality)(snapshot)
    report = task(render_quality)(checked)
    task(enforce_quality)(report)
