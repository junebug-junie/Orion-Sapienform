FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY services/orion-gpu-pool/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
# The pool's whole rule set, and the model catalog it resolves discovered profiles against.
COPY config/gpu_pool.yaml /app/config/gpu_pool.yaml
COPY config/llm_profiles.yaml /app/config/llm_profiles.yaml
# The visual_baseline swap guard reads the same policy the visual chain runs on (interval, freshness).
COPY config/proposals/visual_baseline.v1.yaml /app/config/proposals/visual_baseline.v1.yaml
COPY services/orion-gpu-pool /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8127"]
