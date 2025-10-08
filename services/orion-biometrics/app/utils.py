import subprocess
import csv
import os
import time
from datetime import datetime

# ✅ Path to where timestamped GPU telemetry CSVs are saved
TELEMETRY_DIR = "/mnt/telemetry/gpu_stats"
LOG_FILE = "/app/logs/error.log"

def collect_gpu_stats():
    """
    Collects the latest GPU stats by:

    1. Running a shell script that writes a fresh GPU CSV to /mnt/telemetry/gpu_stats/
    2. Waiting briefly to ensure the file is written.
    3. Reading the most recent file's contents and returning as dict.
    4. Logging any failure to persistent Docker volume.
    """
    try:
        # 🔁 Run the telemetry shell script (writes a new timestamped CSV)
        subprocess.run(["/orion/sensors/gpu_host_stats.sh"], check=True)

        # ⏳ Ensure OS has time to write the file
        time.sleep(1)

        # 📂 Find most recent .csv file in telemetry directory
        files = [f for f in os.listdir(TELEMETRY_DIR) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError("No GPU CSV files found in telemetry directory.")

        latest_file = max(
            files,
            key=lambda f: os.path.getmtime(os.path.join(TELEMETRY_DIR, f))
        )

        # 📖 Read the latest CSV
        full_path = os.path.join(TELEMETRY_DIR, latest_file)
        with open(full_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        return {
            "latest_file": latest_file,
            "gpus": rows
        }

    except Exception as e:
        # ❌ Persist error logs
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as log:
            log.write(f"[{datetime.now().isoformat()}] GPU read error: {e}\n")

        return {"gpus": [], "error": str(e)}
