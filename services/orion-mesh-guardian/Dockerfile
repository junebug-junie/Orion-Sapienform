# services/orion-mesh-guardian/Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-mesh-guardian/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY services/orion-mesh-guardian/app ./app
COPY orion ./orion
COPY config ./config

EXPOSE 7161

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7161"]
