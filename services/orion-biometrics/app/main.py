from fastapi import FastAPI
import uvicorn
import redis

from app.settings import settings
from orion.core.bus import OrionBus

from datetime import datetime
import subprocess
import json

app = FastAPI(settings.SERVICE_NAME, settings.SERVICE_VERSION)
bus = OrionBus(url=settings.ORION_BUS_URL)

def collect_biometrics():
    # GPU metrics
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv"])

    # Sensors for CPU temperature
    cpu_info = subprocess.check_output(["sensors", "-j"])

    # Parse and structure data
    gpu_data = gpu_info.decode('utf-8').strip()
    cpu_data = json.loads(cpu_info)

    # Combine into a single dict
    biometrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu_data,
        "cpu": cpu_data
    }

    return biometrics

@app.get("/latest-biometrics")
def get_latest_biometrics():

    # Collect the current biometrics
    biometrics = collect_biometrics()
    return biometrics

def publish_biometrics_to_db(biometrics):
    # Use Orion SQL writer client to write to PostgreSQL
    # Placeholder for actual client code
    pass  # Implement your DB writing logic here
    bus = OrionBus(url=settings.ORION_BUS_URL)
