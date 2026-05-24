#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== unit tests (host) =="
PYTHONPATH="services/orion-biometrics:." pytest \
  services/orion-biometrics/tests/test_node_catalog.py \
  services/orion-biometrics/tests/test_biometrics_grammar_emit.py -q

echo "== optional: container tests =="
if docker compose -f services/orion-biometrics/docker-compose.yml ps -q biometrics 2>/dev/null | grep -q .; then
  docker compose -f services/orion-biometrics/docker-compose.yml exec -T biometrics \
    pytest tests/test_node_catalog.py tests/test_biometrics_grammar_emit.py -q
fi

echo "== bus tap (manual, 30s) =="
echo "Run: redis-cli SUBSCRIBE orion:grammar:event"
echo "Expect: schema_version grammar_event.v1, trace_id biometrics.node:<node>:..., provenance.source_service orion-biometrics"
