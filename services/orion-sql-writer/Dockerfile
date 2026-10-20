FROM python:3.12-slim

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY services/orion-sql-writer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/orion-sql-writer/app ./app

# copy schemas into the container
COPY orion /app/orion

# Camera zones (patio no-embed rule) for the walkway reducers; resolved by
# orion/vision/zones.py as /app/config/vision_zones.yaml.
COPY config/vision_zones.yaml /app/config/vision_zones.yaml
# Camera portals for the handoff reducer (orion/vision/topology.py).
COPY config/vision_topology.yaml /app/config/vision_topology.yaml

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8220"]
