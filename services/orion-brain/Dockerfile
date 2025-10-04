# Dockerfile
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY services/orion-brain/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy shared code
COPY orion /app/orion

# Copy the service code
COPY services/orion-brain /app

ENV PORT=8088
EXPOSE 8088
CMD ["sh","-lc","python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8088} --log-level info"]
