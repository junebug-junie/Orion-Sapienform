# services/orion-biometrics/Dockerfile

FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# OS deps + Python (3.12 is native on 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      lm-sensors pciutils curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

# Use a venv to avoid PEP 668 (externally-managed)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Python deps
COPY services/orion-biometrics/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# App code
COPY services/orion-biometrics/app ./app
COPY orion /app/orion

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD nvidia-smi -L >/dev/null 2>&1 || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--log-level", "info"]
