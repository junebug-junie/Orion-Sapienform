#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TOPIC_FOUNDRY_BASE_URL:-http://127.0.0.1:8615}"
SOURCE_TABLE="${TOPIC_FOUNDRY_SOURCE_TABLE:-public.smoke_topic_foundry}"
ID_COLUMN="${TOPIC_FOUNDRY_ID_COLUMN:-id}"
TIME_COLUMN="${TOPIC_FOUNDRY_TIME_COLUMN:-created_at}"
TEXT_COLUMNS_CSV="${TOPIC_FOUNDRY_TEXT_COLUMNS:-body}"
LIMIT="${TOPIC_FOUNDRY_PREVIEW_LIMIT:-500}"

readarray -t TEXT_COLUMNS < <(python - <<'PY' "$TEXT_COLUMNS_CSV"
import json,sys
cols=[c.strip() for c in sys.argv[1].split(',') if c.strip()]
print('\n'.join(cols))
PY
)

payload_create="$(python - <<'PY' "$SOURCE_TABLE" "$ID_COLUMN" "$TIME_COLUMN" "${TEXT_COLUMNS[@]}"
import json,sys
source_table=sys.argv[1]
id_col=sys.argv[2]
time_col=sys.argv[3]
text_cols=sys.argv[4:]
print(json.dumps({
  'name':'smoke_preview_doc_counts',
  'source_table':source_table,
  'id_column':id_col,
  'time_column':time_col,
  'text_columns':text_cols,
  'boundary_column':None,
  'boundary_strategy':None,
}))
PY
)"

create_body="$(mktemp)"
create_status="$(curl -sS -o "$create_body" -w "%{http_code}" -X POST "$BASE_URL/datasets" -H 'content-type: application/json' -d "$payload_create")"
if [[ "$create_status" != "200" && "$create_status" != "201" ]]; then
  echo "[FAIL] dataset create returned HTTP $create_status"
  cat "$create_body"
  exit 1
fi

DATASET_ID="$(python - <<'PY' "$create_body"
import json,sys
print(json.load(open(sys.argv[1])).get('dataset_id',''))
PY
)"
if [[ -z "$DATASET_ID" ]]; then
  echo "[FAIL] dataset_id missing"
  cat "$create_body"
  exit 1
fi

preview_payload="$(python - <<'PY' "$DATASET_ID" "$LIMIT"
import json,sys
print(json.dumps({
  'dataset_id': sys.argv[1],
  'limit': int(sys.argv[2]),
  'windowing': {
    'windowing_mode': 'document',
    'time_gap_minutes': 15,
    'max_chars': 6000,
  }
}))
PY
)"

preview_body="$(mktemp)"
preview_status="$(curl -sS -o "$preview_body" -w "%{http_code}" -X POST "$BASE_URL/datasets/preview" -H 'content-type: application/json' -d "$preview_payload")"
if [[ "$preview_status" != "200" && "$preview_status" != "201" ]]; then
  echo "[FAIL] preview returned HTTP $preview_status"
  cat "$preview_body"
  exit 1
fi

read -r ROW_COUNT DOC_COUNT SEGMENT_COUNT < <(python - <<'PY' "$preview_body"
import json,sys
obj=json.load(open(sys.argv[1]))
row_count=obj.get('row_count', obj.get('rows_scanned', 0))
doc_count=obj.get('doc_count', obj.get('docs_generated', 0))
segment_count=obj.get('segment_count', obj.get('segments_generated', 0))
print(row_count, doc_count, segment_count)
PY
)

echo "[INFO] dataset_id=$DATASET_ID row_count=$ROW_COUNT doc_count=$DOC_COUNT segment_count=$SEGMENT_COUNT"

if (( ROW_COUNT < 50 )); then
  echo "[FAIL] expected row_count >= 50, got $ROW_COUNT"
  cat "$preview_body"
  exit 1
fi

if (( DOC_COUNT < 50 )); then
  echo "[FAIL] expected doc_count >= 50 in document mode, got $DOC_COUNT"
  cat "$preview_body"
  exit 1
fi

echo "[PASS] document preview doc_count follows row_count (>=50)"
