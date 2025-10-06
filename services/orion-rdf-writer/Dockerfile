FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY services/orion-rdf-writer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/orion-rdf-writer/app ./app

# copy schemas into the container
COPY orion /app/orion

# add orion/ to PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020", "--log-level", "info"]
