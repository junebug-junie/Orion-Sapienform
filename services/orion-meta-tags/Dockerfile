FROM python:3.12-slim

WORKDIR /app

# Install deps# Install deps
COPY services/orion-meta-tags/requirements.txt .
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_trf


# Copy bus client (relative path inside monorepo)
#COPY services/orion-bus/client /tmp/orion-bus-client
#RUN pip3 install /tmp/orion-bus-client

# Copy code
COPY services/orion-meta-tags/app ./app

# Copy shared orion libs (schemas, etc.)
COPY orion /app/orion

# Add to PYTHONPATH so `from orion.schemas ...` works
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8201

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8201"]
