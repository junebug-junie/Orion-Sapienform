FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-social-memory/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY services/orion-social-memory/app ./app

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8765

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
