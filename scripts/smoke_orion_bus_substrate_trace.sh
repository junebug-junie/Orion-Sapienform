#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

echo "=== 1. Unit tests ==="
PYTHONPATH=services/orion-bus:. "$PY" -m pytest \
  services/orion-bus/tests/test_orion_bus_grammar_emit.py \
  services/orion-bus/tests/test_orion_bus_grammar_publish_fail_open.py \
  services/orion-bus/tests/test_orion_bus_observer_rollup.py \
  -q

echo "=== 2. Compileall ==="
PYTHONPATH=. "$PY" -m compileall services/orion-bus -q

echo "=== 3. Docker bus containers ==="
docker ps --format 'table {{.Names}}\t{{.Image}}' | grep -Ei 'bus' || true

OBSERVER="$(docker ps --format '{{.Names}}' | grep -E 'bus-observer' | head -1 || true)"
if [ -n "${OBSERVER}" ]; then
  echo "=== 4. Observer logs (tail 200) ==="
  docker logs --tail=200 "${OBSERVER}" 2>&1 | grep -Ei 'bus|trace|grammar|failed|error' || true
else
  echo "WARN: no bus-observer container found; start with: cd services/orion-bus && make up"
fi

echo "=== 5. SQL proof (run against grammar_events DB) ==="
cat <<'SQL'
select
    created_at
  , source_service
  , trace_id
  , event_json::jsonb #>> '{atom,semantic_role}' as semantic_role
  , event_json::jsonb #>> '{atom,summary}' as summary
from grammar_events
where source_service in ('orion-bus', 'orion-bus-tap')
  and trace_id like 'bus.transport:%'
order by created_at desc
limit 30;
SQL
