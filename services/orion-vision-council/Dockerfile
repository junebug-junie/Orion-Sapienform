FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/orion-vision-council/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-vision-council/app ./app
COPY orion ./orion
# Camera zones (orion/vision/zones.py resolves /app/config/vision_zones.yaml).
# app/unresolved.py reads them so a walkway percept never records patio
# content; if this file is missing every stream fails closed (no free text,
# no frames), so it must ship in the image.
COPY config/vision_zones.yaml /app/config/vision_zones.yaml

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
