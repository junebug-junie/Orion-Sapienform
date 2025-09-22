# Dockerfile
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY . /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txthttpx==0.27.2 uvloop==0.20.0 redis==5.0.7

ENV PORT=8088
EXPOSE 8088
CMD ["sh","-lc","python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8088} --log-level info"]
