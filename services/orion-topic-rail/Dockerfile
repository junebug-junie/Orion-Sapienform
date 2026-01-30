FROM python:3.12-slim

WORKDIR /app

# Better runtime behavior for logs + crashes
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
# - build-essential: needed to compile hdbscan (C/Cython extension)
# - curl/jq: for debugging and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      jq \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY services/orion-topic-rail/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Optionally drop build toolchain after wheels are built (keeps curl/jq)
RUN apt-get purge -y --auto-remove build-essential || true

# Copy service code
COPY services/orion-topic-rail/app ./app

# Copy shared monorepo package(s) the service imports
COPY orion ./orion

ENV PYTHONPATH=/app

CMD ["python", "-m", "app.main"]
