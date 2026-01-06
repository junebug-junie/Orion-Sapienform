FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-equilibrium-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY services/orion-equilibrium-service /app

CMD ["python", "-m", "app.main"]
