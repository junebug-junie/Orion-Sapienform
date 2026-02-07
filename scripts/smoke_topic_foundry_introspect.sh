#!/usr/bin/env bash
set -euo pipefail

TOPIC_FOUNDRY_URL=${TOPIC_FOUNDRY_URL:-http://localhost:8615}

schemas_json=$(curl -sS "${TOPIC_FOUNDRY_URL}/introspect/schemas")
first_schema=$(python3 - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    schemas=data.get('schemas') or []
    print(schemas[0] if schemas else '')
except Exception:
    print('')
PY
<<<"$schemas_json")

printf "schemas status: %s\n" "$(python3 - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    print(len(data.get('schemas') or []))
except Exception:
    print('error')
PY
<<<"$schemas_json")"

if [ -z "$first_schema" ]; then
  echo "No schemas returned; skipping tables/columns."
  exit 0
fi

tables_json=$(curl -sS "${TOPIC_FOUNDRY_URL}/introspect/tables?schema=${first_schema}")
first_table=$(python3 - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    tables=data.get('tables') or []
    print(tables[0]['table_name'] if tables else '')
except Exception:
    print('')
PY
<<<"$tables_json")

printf "tables status: %s\n" "$(python3 - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    print(len(data.get('tables') or []))
except Exception:
    print('error')
PY
<<<"$tables_json")"

if [ -z "$first_table" ]; then
  echo "No tables returned; skipping columns."
  exit 0
fi

columns_json=$(curl -sS "${TOPIC_FOUNDRY_URL}/introspect/columns?schema=${first_schema}&table=${first_table}")
printf "columns status: %s\n" "$(python3 - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    print(len(data.get('columns') or []))
except Exception:
    print('error')
PY
<<<"$columns_json")"

