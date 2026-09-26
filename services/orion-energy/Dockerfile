FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/orion-energy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-energy /app
COPY orion /app/orion
COPY config/energy /app/config/energy

CMD ["python", "-m", "app.main"]
