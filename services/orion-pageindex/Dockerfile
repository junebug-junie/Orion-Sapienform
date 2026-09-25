FROM python:3.11-slim

WORKDIR /app

ARG PAGEINDEX_REPO_URL=https://github.com/VectifyAI/PageIndex.git
ARG PAGEINDEX_REF=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The Orion service (FastAPI/uvicorn) lives in its own venv, isolated from PageIndex.
# Root cause of the 2026-08-30 crash loop (RestartCount ~4.8k): PageIndex's upstream
# requirements (unpinned `main`, `mcp>=1.19.0,<3`) were installed into the SAME interpreter
# after ours and silently upgraded starlette 0.41 -> 1.6 (and pydantic 2.10 -> 2.13) under our
# pinned fastapi==0.115.6. starlette 1.x dropped Router(on_startup=...), so `app.main` died at
# import with `Router.__init__() got an unexpected keyword argument 'on_startup'`.
# PageIndex only ever runs as a subprocess (`PAGEINDEX_PYTHON_BIN=python3`, see
# app/pageindex_cli.py), so it keeps the system interpreter and its deps can no longer reach ours.
ENV ORION_VENV=/opt/orion-venv
COPY services/orion-pageindex/requirements.txt /tmp/requirements.txt
RUN python -m venv ${ORION_VENV} \
    && ${ORION_VENV}/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && ${ORION_VENV}/bin/pip check
RUN git clone --depth 1 --branch ${PAGEINDEX_REF} ${PAGEINDEX_REPO_URL} /opt/PageIndex \
    && sed -i 's/^python-dotenv==1\.2\.2$/python-dotenv==1.0.1/' /opt/PageIndex/requirements.txt \
    && pip install --no-cache-dir -r /opt/PageIndex/requirements.txt

COPY services/orion-pageindex /app
# Real blocker found while wiring the bus-native SystemHealthV1 heartbeat (see
# docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md): this Dockerfile
# never copied the shared `orion` package in, so `orion.core.bus.bus_service_chassis` was not
# importable at runtime despite being importable from a repo-root checkout during dev/tests.
COPY orion /app/orion

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Build-time guard: fail the BUILD (not a restart loop) if the service venv cannot import the app
# after PageIndex's deps have landed in the system interpreter.
RUN ${ORION_VENV}/bin/python -c "import fastapi, starlette; fastapi.APIRouter(); import app.main; print('app.main import ok: fastapi', fastapi.__version__, 'starlette', starlette.__version__)"
# ...and the PageIndex side must still run under the system interpreter the service shells out to.
RUN cd /opt/PageIndex && python3 run_pageindex.py --help >/dev/null && echo "PageIndex CLI ok under $(command -v python3)"

# Explicit venv path (PATH is deliberately NOT changed, so `python3` for the PageIndex CLI
# subprocess still resolves to the system interpreter that has PageIndex installed).
CMD ["/opt/orion-venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8360", "--log-level", "info"]
