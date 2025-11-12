# orion_biometrics/metrics.py
import subprocess
import json
from datetime import datetime
from app.settings import settings
from app.utils import collect_gpu_stats

def collect_biometrics():
    try:
        # GPU metrics
        gpu_data = collect_gpu_stats()

    except Exception as e:
        gpu_data = f"Error: {e}"

    try:
        # CPU metrics via sensors
        cpu_info = subprocess.check_output(["sensors", "-j"])
        cpu_data = json.loads(cpu_info)
    except Exception as e:
        cpu_data = {"error": str(e)}

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu_data,
        "cpu": cpu_data,
        "node": settings.NODE_NAME,
        "service_name": settings.SERVICE_NAME,
        "service_version": settings.SERVICE_VERSION,
    }
