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
# Body economy Patch B2: heat sensitivities as data (written by fit_heat_sensitivity.py --write-config).
COPY config/physiology/heat_sensitivity.yaml /app/config/physiology/heat_sensitivity.yaml
# Body economy Patch C1: want families (tier, prior weight, deferrable window, lanes) for /physiology/bid.
COPY config/physiology/families.yaml /app/config/physiology/families.yaml
# Pinned per-host rest watts: GET /physiology/report hosts/unit_costs measure above this (PR #2992).
COPY config/physiology/rest_body.yaml /app/config/physiology/rest_body.yaml
COPY services/orion-durable-runs /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8121"]
