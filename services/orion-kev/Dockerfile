# Kev / TypeSafe-compatible System One endpoint for Orion Athena.
# Serves POST /v1/systemone on :8009. Consumed by orion-substrate-runtime
# shadow appraisal (SUBSTRATE_SYSTEM_ONE_BASE_URL=http://orion-athena-kev:8009).
#
# Base image is Python 3.11; upstream kev pyproject wants >=3.12, so we do not
# `pip install .` — PYTHONPATH=/app and run the module directly.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    KEV_MODEL=jaredpalmer/kev-0.8b

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Pin by SHA so rebuilds are reproducible. Override: --build-arg KEV_GIT_SHA=...
ARG KEV_GIT_SHA=557598fced1dada75dfbf36ed144dce309ac6ceb
RUN git clone https://github.com/jaredpalmer/kev.git /tmp/kev \
 && cd /tmp/kev && git checkout "${KEV_GIT_SHA}" \
 && cp -a /tmp/kev/kev /app/kev \
 && rm -rf /tmp/kev \
 && sed -i 's/host="127.0.0.1"/host="0.0.0.0"/' /app/kev/serve.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN pip install --no-cache-dir \
      "fastapi>=0.115" \
      "uvicorn>=0.30" \
      "typesafe-sdk>=0.6.0" \
      "accelerate>=1.15.0" \
      "peft>=0.21" \
      "transformers>=5.17,<6" \
      "pydantic>=2.9" \
      "safetensors>=0.4" \
      "huggingface_hub>=0.26"

EXPOSE 8009
ENTRYPOINT ["/entrypoint.sh"]
