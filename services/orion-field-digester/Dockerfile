FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-field-digester/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY config/field /app/config/field
# 2026-08-16, live-caught same-day as deploy: PR #1699's app/digestion/
# tension.py::update_tension_pressure() loads config/attention/
# channel_direction_map.yaml (orion.attention.tension.direction_map's
# _DEFAULT_PATH is Docker-mount-relative, not repo-root-relative -- this
# service does not mount /repo the way Hub's Dockerfile does, so a
# path-candidate fallback wasn't the fix here; the file must actually be in
# the image). Without this line every digestion tick raised
# DirectionMapError uncaught through run_digestion_tick(), which took down
# the WHOLE tick -- decay/diffusion/suppression/precision-baseline update
# and the Postgres write all never ran, not just the new tension step.
# substrate_field_state was frozen for the entire time between deploy and
# this fix. Same missing-config-directory shape channel_glossary.py's own
# docstring already documents for Hub's Dockerfile -- caught live here
# instead of by that same class of bug repeating a third time.
COPY config/attention /app/config/attention
COPY services/orion-field-digester /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8116"]
