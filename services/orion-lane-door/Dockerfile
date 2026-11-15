# The lane door: stdlib-only Python, no pip install. Build context is the repo root (compose: ../..).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/lane-door
COPY services/orion-lane-door/app ./app
# Host network: the door listens on its lane port (LANE_DOOR_LISTEN_PORT), not a container port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "-m", "app.health"]
CMD ["python", "-m", "app.main"]
