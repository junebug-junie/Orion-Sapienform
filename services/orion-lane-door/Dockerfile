# The lane door: stdlib-only Python, no pip install. Build context is the repo root (compose: ../..).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/lane-door
COPY services/orion-lane-door/app ./app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/door/stats', timeout=4)"
CMD ["python", "-m", "app.main"]
