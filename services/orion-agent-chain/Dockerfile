# ─────────────────────────────────────────────
# 🤖 Orion Agent Chain — container
# ─────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps if needed (curl for debug, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements early for layer caching
# (explicit path since build context = repo root)
COPY services/orion-planner-react/requirements.txt ./requirements.txt

# Install service dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy core Orion package + this service code.
# Assumes build context is repo root and 'orion' is the core package.
COPY orion /app/orion
COPY services/orion-agent-chain/app /app/app

EXPOSE 8092

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8092"]
