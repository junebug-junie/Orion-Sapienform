FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first
COPY services/orion-collapse-mirror/requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY services/orion-collapse-mirror/app ./app

# Copy shared orion libs (schemas, etc.)
COPY orion /app/orion

# Add to PYTHONPATH so `from orion.schemas ...` works
ENV PYTHONPATH="/app:${PYTHONPATH}"

# (Optional) install schemas as editable if you want pip metadata
# RUN pip install -e /app/orion/schemas

# Pre-pull embedding model into container (so no runtime downloads)
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose consistent port
EXPOSE 8087

# Run via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8087", "--reload"]
