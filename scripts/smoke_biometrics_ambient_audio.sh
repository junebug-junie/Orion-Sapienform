#!/usr/bin/env bash
# Smoke: ambient audio snapshot -> biometrics sample/summary/grammar.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SNAP="${ORION_AMBIENT_AUDIO_PATH:-/run/orion-audio/latest.json}"
BIO_URL="${BIOMETRICS_URL:-http://127.0.0.1:8100}"
PY="${PYTHON:-/mnt/scripts/Orion-Sapienform/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

export PYTHONPATH="${ROOT}/services/orion-biometrics:${ROOT}${PYTHONPATH:+:}${PYTHONPATH:-}"

echo "== unit: ambient audio snapshot + grammar + pressure =="
"${PY}" -m pytest \
  services/orion-biometrics/tests/test_ambient_audio_snapshot.py \
  services/orion-biometrics/tests/test_ambient_audio_grammar.py \
  tests/test_ambient_audio.py \
  -q

echo "== ambient audio snapshot path: ${SNAP} =="
"${PY}" - <<'PY' "${SNAP}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
assert path.is_file(), f"snapshot missing at {path}"
data = json.loads(path.read_text())
assert data.get("schema") == "orion.ambient_audio.v1", data
assert data.get("status") in {"ok", "stale", "error", "missing"}, data
if data.get("status") == "ok":
    assert data.get("rms") is not None, data
    assert data.get("peak") is not None, data
print(f"OK snapshot status={data.get('status')} rms={data.get('rms')} peak={data.get('peak')}")
PY

echo "== biometrics health =="
curl -fsS "${BIO_URL}/health" | "${PY}" -c 'import json,sys; print(json.load(sys.stdin))'

echo "== biometrics recent ambient sample =="
curl -fsS "${BIO_URL}/raw/recent?limit=20" | "${PY}" -c '
import json, sys
items = json.load(sys.stdin).get("items", [])
for item in items:
    audio = (item.get("sample") or {}).get("ambient_audio")
    if audio is not None:
        assert "rms" in audio and "peak" in audio, audio
        print("OK sample.ambient_audio present:", audio)
        break
else:
    raise SystemExit("FAIL: no recent sample.ambient_audio; check reader and bind mount")
'

echo "== biometrics ambient summary pressure =="
curl -fsS "${BIO_URL}/snapshot" | "${PY}" -c '
import json, sys
nodes = json.load(sys.stdin).get("nodes", {})
for node_id, node in nodes.items():
    pressures = ((node or {}).get("summary") or {}).get("pressures", {})
    if "cabinet_ambient_audio_activity" in pressures:
        print(f"OK {node_id} ambient activity pressure={pressures['"'"'cabinet_ambient_audio_activity'"'"']}")
        break
else:
    raise SystemExit("FAIL: no cabinet_ambient_audio_activity in live node summaries")
'

echo "== live ambient grammar publication =="
"${PY}" - <<'PY' "${BIO_URL}"
import json
import sys
import time
from urllib.request import urlopen

import redis

base_url = sys.argv[1].rstrip("/")
with urlopen(f"{base_url}/health", timeout=5) as response:
    health = json.load(response)
assert health.get("publish_biometrics_grammar") is True, health

roles_needed = {
    "cabinet_ambient_audio_activity_signal",
    "cabinet_ambient_audio_staleness_signal",
}
roles_seen = set()
client = redis.Redis.from_url(health["bus_url"], decode_responses=True)
pubsub = client.pubsub(ignore_subscribe_messages=True)
pubsub.subscribe(health["grammar_event_channel"])
deadline = time.monotonic() + 45.0
try:
    while time.monotonic() < deadline and roles_seen != roles_needed:
        message = pubsub.get_message(timeout=1.0)
        if not message or message.get("type") != "message":
            continue
        raw = message.get("data")
        try:
            envelope = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        payload = envelope.get("payload") if isinstance(envelope, dict) else None
        atom = payload.get("atom") if isinstance(payload, dict) else None
        role = atom.get("semantic_role") if isinstance(atom, dict) else None
        if role in roles_needed:
            roles_seen.add(role)
finally:
    pubsub.close()
    client.close()

missing = sorted(roles_needed - roles_seen)
assert not missing, f"missing live grammar roles after 45s: {missing}"
print("OK live grammar roles:", sorted(roles_seen))
PY

echo "smoke_biometrics_ambient_audio: OK"
