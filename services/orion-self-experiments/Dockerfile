FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-self-experiments/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY services/orion-self-experiments/app ./app
COPY orion ./orion

EXPOSE 7172

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7172"]
