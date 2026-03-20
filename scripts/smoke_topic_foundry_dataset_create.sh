#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TOPIC_FOUNDRY_BASE_URL:-http://127.0.0.1:8615}"

payload='{"name":"smoke_dataset","source_table":"public.smoke_topic_foundry","id_column":"id","time_column":"created_at","text_columns":["body"],"boundary_column":null,"boundary_strategy":null}'

resp_file="$(mktemp)"
status="$(curl -sS -o "$resp_file" -w "%{http_code}" -X POST "$BASE_URL/datasets" -H "content-type: application/json" -d "$payload")"

if [[ "$status" != "200" && "$status" != "201" ]]; then
  echo "[FAIL] POST $BASE_URL/datasets returned HTTP $status"
  cat "$resp_file"
  exit 1
fi

dataset_id="$(python - <<'PY' "$resp_file"
import json,sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data=json.load(f)
print(data.get('dataset_id',''))
PY
)"

if [[ -z "$dataset_id" ]]; then
  echo "[FAIL] dataset_id missing in response"
  cat "$resp_file"
  exit 1
fi

list_resp="$(curl -sS "$BASE_URL/datasets")"
timezone="$(python - <<'PY' "$list_resp" "$dataset_id"
import json,sys
items=json.loads(sys.argv[1]).get("datasets", [])
want=sys.argv[2]
for item in items:
    if item.get("dataset_id") == want:
        print(item.get("timezone", ""))
        break
PY
)"

if [[ "$timezone" != "UTC" ]]; then
  echo "[FAIL] expected timezone UTC for dataset_id=$dataset_id, got '$timezone'"
  echo "$list_resp"
  exit 1
fi

echo "[PASS] created dataset_id=$dataset_id timezone=$timezone"
