#!/usr/bin/env bash
# Smoke: biometrics sees a fresh cabinet sensor snapshot (file or live reader).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SNAP="${ORION_CABINET_SENSORS_PATH:-/run/orion-sensors/latest.json}"
BIO_URL="${BIOMETRICS_URL:-http://127.0.0.1:8100}"

echo "== cabinet snapshot path: ${SNAP} =="
if [[ ! -f "${SNAP}" ]]; then
  echo "FAIL: snapshot missing at ${SNAP}"
  echo "Hint: run scripts/setup_athena_cabinet_sensors.sh and start orion-cabinet-sensors.service"
  exit 1
fi

python3 - <<'PY' "${SNAP}"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
status = data.get("status")
frame = data.get("frame")
assert status in {"ok", "stale", "error", "missing"}, data
assert isinstance(frame, dict) or status != "ok", data
if status == "ok":
    assert frame.get("schema") == "orion.sensor_frame.v1", frame
    print(f"OK snapshot status=ok schema={frame.get('schema')} seq={frame.get('seq')}")
else:
    print(f"WARN snapshot status={status} (reader up but not ok)")
PY

echo "== biometrics health =="
curl -fsS "${BIO_URL}/health" | python3 -c 'import json,sys; print(json.load(sys.stdin))'

echo "== biometrics recent sample (sensors key) =="
# Prefer a debug/recent endpoint if present; else instruct redis tap.
if curl -fsS "${BIO_URL}/debug/recent" >/tmp/bio_recent.json 2>/dev/null; then
  python3 - <<'PY'
import json
from pathlib import Path
raw = json.loads(Path("/tmp/bio_recent.json").read_text())
# tolerate list or dict envelopes
items = raw if isinstance(raw, list) else raw.get("samples") or raw.get("recent") or [raw]
found = False
for item in items[:20]:
    payload = item.get("payload") if isinstance(item, dict) else None
    sample = payload if isinstance(payload, dict) else item if isinstance(item, dict) else {}
    sensors = sample.get("sensors") if isinstance(sample, dict) else None
    if sensors:
        print("OK sample.sensors present:", sorted(sensors.keys()) if isinstance(sensors, dict) else type(sensors))
        found = True
        break
if not found:
    print("WARN: /debug/recent had no sample.sensors yet — check bind mount and reader freshness")
    raise SystemExit(1)
PY
else
  echo "WARN: ${BIO_URL}/debug/recent not available"
  echo "Manual check: redis-cli SUBSCRIBE orion:biometrics:sample — expect sensors.frame when Nano is live"
fi

echo "smoke_biometrics_cabinet_sensors: OK"
