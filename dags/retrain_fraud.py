"""Graph assembly only; all task functions live in fraud_pipeline.tasks."""
from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.decorators import task
from fraud_pipeline.tasks.prepare import prepare_window, split_by_time
from fraud_pipeline.tasks.train import train_candidate
from fraud_pipeline.tasks.evaluate import evaluate_candidate
from fraud_pipeline.tasks.publish import publish_candidate

with DAG("retrain_fraud", schedule="1-59/5 * * * *", start_date=datetime(2024,1,1,tzinfo=timezone.utc),
         catchup=False, max_active_runs=1, dagrun_timeout=timedelta(minutes=15),
         default_args={"retries": 2, "retry_delay": timedelta(seconds=20)}, tags=["fraud", "training"]) as dag:
    window = task(prepare_window)(run_id="{{ run_id }}")
    split = task(split_by_time)(window)
    candidate = task(train_candidate)(split)
    evaluated = task(evaluate_candidate)(candidate)
    task(publish_candidate)(evaluated)
