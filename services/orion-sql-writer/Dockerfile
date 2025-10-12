FROM python:3.12-slim

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY services/orion-sql-writer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/orion-sql-writer/app ./app

# copy schemas into the container
COPY orion /app/orion

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8220"]
