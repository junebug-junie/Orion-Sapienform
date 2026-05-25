FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-attention-runtime/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY config/attention /app/config/attention
COPY services/orion-attention-runtime /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8117"]
