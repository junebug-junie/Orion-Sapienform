FROM python:3.12-slim

WORKDIR /app

COPY services/orion-topic-rail/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-topic-rail/app ./app

ENV PYTHONPATH=/app

CMD ["python", "-m", "app.main"]
