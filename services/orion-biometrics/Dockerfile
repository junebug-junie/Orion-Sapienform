# use 3.11 to resolve OpenSSL issues
FROM python:3.12-bookworm

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY services/orion-biometrics/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sensors
RUN apt update && apt install -y lm-sensors pciutils && rm -rf /var/lib/apt/lists/*

# Copy service code
COPY services/orion-biometrics/app ./app

# copy orion scripts into the container
COPY orion /app/orion

# add orion/ to PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--log-level", "info"]
