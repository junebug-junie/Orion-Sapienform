FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    iputils-ping \
    iproute2 \
    dnsutils \
    netcat-traditional \
    procps \
    lsof \
    less \
    vim-tiny \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-landing-pad/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY services/orion-landing-pad /app

EXPOSE 8370

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8370", "--log-level", "info"]
