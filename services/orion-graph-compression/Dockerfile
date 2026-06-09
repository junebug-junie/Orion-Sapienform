FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY services/orion-graph-compression/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY orion /app/orion
COPY services/orion-graph-compression/config /app/config
COPY services/orion-graph-compression/app /app/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8271"]
