FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-durable-runs/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY config/proposals/visual_baseline.v1.yaml /app/config/proposals/visual_baseline.v1.yaml
# app/self_sense_graph.py imports orion.evals.self_sense_runner, which
# transitively imports orion.evals.self_sense -- that module reads this file
# at IMPORT time (module-scope FIELD_NODE_IDS), and self_sense_graph.py is
# itself imported eagerly at DurableRunner.__init__ (building the self-sense-
# eval graph). Without this line the container fails to boot at all the
# first time this workflow is registered -- caught here, before merge, by
# the exact same root cause that crashed orion-hub in production the same
# day (missing this file's copy there too; same fix, see that Dockerfile).
COPY config/field /app/config/field
# Stage 4.5: a run's GPU pool hold is placed by its route's class (orion.gpu_pool.config reads
# /app/config/gpu_pool.yaml: routes -> class, and hold_lease_ttl_sec for the heartbeat check).
COPY config/gpu_pool.yaml /app/config/gpu_pool.yaml
COPY services/orion-durable-runs /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8121"]
