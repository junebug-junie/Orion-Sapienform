import subprocess
import csv
import os
import time
from datetime import datetime

# ✅ Path to where timestamped GPU telemetry CSVs are saved
TELEMETRY_DIR = "/mnt/telemetry/gpu_stats"
LOG_FILE = "/app/logs/error.log"

def _read_gpu_processes(latest_gpu_file):
    """Read the ".procs.csv" sibling of ``latest_gpu_file`` (same timestamp
    stem) and group rows by gpu_uuid. Missing/unreadable file -> {} (never
    raises); this is additive telemetry, not required for the base reading.
    """
    procs_file = latest_gpu_file[: -len(".csv")] + ".procs.csv" if latest_gpu_file.endswith(".csv") else latest_gpu_file + ".procs.csv"
    full_path = os.path.join(TELEMETRY_DIR, procs_file)
    if not os.path.isfile(full_path):
        return {}
    try:
        with open(full_path, newline="") as f:
            reader = csv.DictReader(f)
            by_uuid = {}
            for row in reader:
                uuid = row.get("gpu_uuid", "")
                by_uuid.setdefault(uuid, []).append(
                    {
                        "pid": row.get("pid"),
                        "process_name": row.get("process_name"),
                        "used_memory_mb": row.get("used_memory_mb"),
                    }
                )
            return by_uuid
    except Exception as e:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as log:
            log.write(f"[{datetime.now().isoformat()}] GPU process read error: {e}\n")
        return {}


def collect_gpu_stats():
    """
    Collects the latest GPU stats by:

    1. Running a shell script that writes a fresh GPU CSV to /mnt/telemetry/gpu_stats/
    2. Waiting briefly to ensure the file is written.
    3. Reading the most recent file's contents and returning as dict.
    4. Logging any failure to persistent Docker volume.
    """
    try:
        # 🔁 Run the telemetry shell script (writes a new timestamped CSV,
        # plus a sibling "<ts>.procs.csv" compute-process listing)
        subprocess.run(["/orion/sensors/gpu_host_stats.sh"], check=True)

        # ⏳ Ensure OS has time to write the file
        time.sleep(1)

        # 📂 Find most recent GPU-stats .csv file, explicitly excluding the
        # sibling ".procs.csv" file -- both end in ".csv", so a naive
        # f.endswith(".csv") filter can pick the process listing as the
        # "latest file" and parse it with the wrong schema.
        files = [
            f for f in os.listdir(TELEMETRY_DIR)
            if f.endswith(".csv") and not f.endswith(".procs.csv")
        ]
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

        # 📖 Read the matching process listing, if present, and attach each
        # GPU's process list by gpu_uuid. Absent/unreadable procs file
        # degrades to an empty list per row, never raises -- process capture
        # is additive telemetry, not required for the base GPU reading.
        processes_by_uuid = _read_gpu_processes(latest_file)
        for row in rows:
            uuid = row.get("gpu_uuid", "")
            row["processes"] = processes_by_uuid.get(uuid, [])

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
