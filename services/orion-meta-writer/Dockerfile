FROM python:3.12-slim

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY services/orion-meta-writer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/orion-meta-writer/app ./app

# Install bus client from monorepo and clean up
COPY services/orion-bus/client /tmp/orion-bus-client
RUN pip install --no-cache-dir /tmp/orion-bus-client && rm -rf /tmp/orion-bus-client

EXPOSE 8202

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8202"]
