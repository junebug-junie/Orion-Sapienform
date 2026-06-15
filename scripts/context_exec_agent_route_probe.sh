#!/usr/bin/env bash
# Live probe: Hub Agent mode → context-exec with explicit llm route.
# Usage:
#   HUB_BASE_URL=http://127.0.0.1:8080 \
#   AGENT_ROUTE=chat \
#   AGENT_TEXT="Where did the Denver belief come from?" \
#   bash scripts/context_exec_agent_route_probe.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HUB_BASE="${HUB_BASE_URL:-http://127.0.0.1:8080}"
AGENT_ROUTE="${AGENT_ROUTE:-chat}"
AGENT_TEXT="${AGENT_TEXT:-Where did the Denver belief come from?}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

fail() { echo "PROBE_FAIL: $*" >&2; exit 1; }

echo "== context-exec agent route probe =="
echo "hub=${HUB_BASE} route=${AGENT_ROUTE}"

routes_json="$(curl -sf "${HUB_BASE}/api/llm-routes" 2>/dev/null || echo '{}')"
route_status="$(echo "$routes_json" | python3 -c "
import json, sys
route = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    print('unknown')
    sys.exit(0)
for item in data.get('routes') or []:
    if str(item.get('id') or '').lower() == route:
        print(str(item.get('status') or 'unknown'))
        sys.exit(0)
print('not_configured')
" "$AGENT_ROUTE")"

if [[ "$route_status" == "down" ]]; then
  echo "ROUTE_DOWN: ${AGENT_ROUTE} status=down"
  exit 2
fi

body="$(AGENT_ROUTE="$AGENT_ROUTE" AGENT_TEXT="$AGENT_TEXT" python3 - <<'PY'
import json, os
print(json.dumps({
  "mode": "agent",
  "messages": [{"role": "user", "content": os.environ["AGENT_TEXT"]}],
  "options": {"llm_route": os.environ["AGENT_ROUTE"]},
  "no_write": True,
}))
PY
)"

resp="$(curl -sf --max-time "${HUB_CHAT_TIMEOUT_SEC:-180}" \
  "${HUB_BASE}/api/chat" \
  -H 'content-type: application/json' \
  -H 'X-Orion-No-Write: 1' \
  -d "$body")"

echo "$resp" | python3 -c "
import json, sys
d = json.load(sys.stdin)
route = '${AGENT_ROUTE}'
dbg = d.get('routing_debug') or {}
ctx = d.get('context_exec_run') or {}
runtime = ctx.get('runtime_debug') or dbg
route_used = runtime.get('route_used') or dbg.get('route_used') or dbg.get('llm_profile')
if route_used != route and not runtime.get('fallback_used'):
    raise SystemExit(f'route mismatch: expected {route}, got {route_used!r} dbg={runtime!r}')
text = d.get('llm_response') or ''
if 'Agent run complete' not in text:
    raise SystemExit(f'missing operator inline response: {text[:200]!r}')
if 'Route:' not in text:
    raise SystemExit('missing Route line in operator response')
op = d.get('operator_summary') or (ctx.get('operator_summary') if ctx else None)
if not op:
    meta = (d.get('raw') or {}).get('metadata') or {}
    op = (meta.get('context_exec') or {}).get('operator_summary')
if not op:
    raise SystemExit('missing operator_summary')
if op.get('agent_mode') is None:
    raise SystemExit('operator_summary missing agent_mode')
safety = op.get('safety') or {}
if safety.get('mutation_performed') is True:
    raise SystemExit('mutation_performed must not be true')
if (ctx.get('artifact') or {}).get('mutation_allowed') is True:
    raise SystemExit('artifact mutation_allowed must not be true')
print(f'PROBE_OK route={route_used} mode={op.get(\"agent_mode\")} synthesis={runtime.get(\"model_synthesis_used\")}')
"

echo "context_exec_agent_route_probe PASS route=${AGENT_ROUTE}"
