FROM python:3.11.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fraud_pipeline ./fraud_pipeline
COPY scripts ./scripts
USER 50000:0
CMD ["python", "-m", "fraud_pipeline.stream", "scorer"]
