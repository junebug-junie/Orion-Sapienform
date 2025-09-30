import os, json, requests, glob, sys
from datetime import datetime

SERVICE_URL = os.getenv("SALIENCE_URL", "http://localhost:8091")
INPUT_DIR = os.getenv("MIRROR_DIR", "./mirrors")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./scored")
STATE_FILE = os.getenv("STATE_FILE", "/app/last_cron.json")

MODE = os.getenv("SALIENCE_MODE", "offline")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def score_file(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    eid = data.get("id", os.path.basename(filepath))

    payload = {"id": eid, "text": data.get("summary", ""), "type": data.get("type", None)}

    try:
        resp = requests.post(f"{SERVICE_URL}/salience/{MODE}", json=payload)
        resp.raise_for_status()
        result = resp.json()

        outpath = os.path.join(OUTPUT_DIR, f"{eid}_scored.json")
        with open(outpath, "w") as out:
            json.dump(result, out, indent=2)
        print(f"✓ Scored {eid} in {MODE} mode → {outpath}")
    except Exception as e:
        print(f"✗ Failed {eid}: {e}")

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    for fp in files:
        score_file(fp)

    # Record last run
    status = {
        "mode": MODE,
        "last_run": datetime.utcnow().isoformat() + "Z",
        "scored_files": len(files)
    }
    with open(STATE_FILE, "w") as f:
        json.dump(status, f, indent=2)
    print(f"🌙 Dream cycle complete → {STATE_FILE}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODE = sys.argv[1]
    main()
