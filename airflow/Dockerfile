FROM apache/airflow:2.10.5-python3.11
ENV PYTHONPATH=/opt/project OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY requirements.txt /tmp/fraud-requirements.txt
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /tmp/fraud-requirements.txt
COPY --chown=airflow:root fraud_pipeline /opt/project/fraud_pipeline
COPY --chown=airflow:root dags /opt/airflow/dags
