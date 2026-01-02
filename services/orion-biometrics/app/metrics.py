# orion_biometrics/metrics.py
import subprocess
import json
import logging
from datetime import datetime
from app.settings import settings
from app.utils import collect_gpu_stats

# Use the logger configured in main/settings if available, else default
logger = logging.getLogger("orion-biometrics")

def collect_biometrics():
    # 1. GPU Metrics
    try:
        gpu_data = collect_gpu_stats()
        # Ensure we didn't get None back
        if gpu_data is None:
             gpu_data = {"status": "no_data"}
    except Exception as e:
        # FIX: Return a Dict, not a String, so it satisfies JSON schema
        gpu_data = {"error": str(e), "status": "failed"}

    # 2. CPU / Sensors
    try:
        # Use -j for JSON output from sensors
        # Note: If sensors hangs, this whole thread hangs. 
        # Ideally run with a timeout if your python version supports it (capture_output=True, timeout=2)
        sensors = subprocess.run(
            ["sensors", "-j"], 
            capture_output=True, 
            text=True,
            timeout=2  # Safety timeout prevents hanging
        )
        cpu_data = json.loads(sensors.stdout)
    except subprocess.TimeoutExpired:
        logger.error("Sensors command timed out")
        cpu_data = {"error": "command_timed_out"}
    except Exception as e:
        logger.error(f"Failed to collect cpu stats: {e}")
        cpu_data = {"error": str(e)}

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu_data,
        "cpu": cpu_data,
        "node": settings.NODE_NAME,
        "service_name": settings.SERVICE_NAME,
        "service_version": settings.SERVICE_VERSION
    }
