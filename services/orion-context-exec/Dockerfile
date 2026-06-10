FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-context-exec/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY services/orion-context-exec/app /app/app

EXPOSE 8096

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8096"]
