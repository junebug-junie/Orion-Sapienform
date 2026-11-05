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
# People Orion knows by name and stated pronouns (app/person_record.py,
# orion/person_model/person.py resolves it as /app/config/person_model/people.yaml).
# people.yaml, plus the camera priors: the committed example and, when the build
# checkout has it, its gitignored camera_priors.local.yaml (orion/person_model/camera_priors.py).
# Copied by folder, not by glob: a wildcard COPY makes BuildKit walk the whole repo,
# which failed on a root-owned scripts/backup/__pycache__ (2026-11-05). Other personal
# *.local.yaml files in the folder are kept out by Dockerfile.dockerignore.
# A changed camera_priors.local.yaml takes effect after a rebuild of orion-sql-writer.
COPY config/person_model/ /app/config/person_model/
# Act energy cost (orion/grammar/act_energy_fill.py): card -> host/nvidia-smi index, and the heat fit.
COPY config/gpu_pool.yaml /app/config/gpu_pool.yaml
COPY config/physiology/heat_sensitivity.yaml /app/config/physiology/heat_sensitivity.yaml
# Pinned per-host rest watts (orion/physiology/rest_body.py): act energy + hourly attribution.
COPY config/physiology/rest_body.yaml /app/config/physiology/rest_body.yaml

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8220"]
