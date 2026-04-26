FROM python:3.11-slim

WORKDIR /app

COPY services/orion-pageindex/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/orion-pageindex /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8360", "--log-level", "info"]
