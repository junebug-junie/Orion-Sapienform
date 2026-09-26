FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        iproute2 \
        iputils-ping \
        dnsutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/orion-zwave/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-zwave /app
COPY orion /app/orion

CMD ["python", "-m", "app.main"]
