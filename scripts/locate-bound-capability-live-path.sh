#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

require_match() {
  local label="$1"
  local file="$2"
  local pattern="$3"
  local out
  out="$(rg -n "$pattern" "$file" || true)"
  if [[ -z "$out" ]]; then
    echo "MISSING: ${label} (${file} :: ${pattern})"
    return 1
  fi
  echo "=== ${label} ==="
  echo "$out"
  echo
}

status=0

require_match "Ingress HTTP endpoint" "services/orion-hub/scripts/api_routes.py" "@router.post\(\"/api/chat\"\)|async def api_chat|async def handle_chat_request" || status=1
require_match "Hub bus egress to gateway" "services/orion-hub/scripts/bus_clients/cortex_client.py" "cortex\.gateway\.chat\.request|CORTEX_GATEWAY_REQUEST_CHANNEL|send_chat_request" || status=1
require_match "Gateway intake + orch handoff" "services/orion-cortex-gateway/app/bus_client.py" "async def handle_gateway_request|rpc_call_cortex_orch|cortex\.orch\.request" || status=1
require_match "Orch handle + verb runtime handoff" "services/orion-cortex-orch/app/main.py" "async def handle\(|call_verb_runtime\(" || status=1
require_match "Orch->Exec publish/wait" "services/orion-cortex-orch/app/orchestrator.py" "async def call_verb_runtime|orion:verb:request|orch_publish_verb_runtime|orch_wait_verb_runtime" || status=1
require_match "Exec verb request entry" "services/orion-cortex-exec/app/main.py" "async def handle_verb_request|verb_runtime_intake|verb_runtime_result" || status=1
require_match "Exec supervisor path + AgentChainService" "services/orion-cortex-exec/app/supervisor.py" "bound_capability_request_received|dispatch_action.*AgentChainService|selected_verb_preserved" || status=1
require_match "Agent-chain bound execution + preserved selected_verb" "services/orion-agent-chain/app/api.py" "bound_capability_direct_execute=1|selected_verb_preserved=1|bound_capability_execution_timeout" || status=1
require_match "Agent-chain concrete skills.runtime invocation" "services/orion-agent-chain/app/tool_executor.py" "_execute_capability_backed_verb|decision\.selected_skill|orion:cortex:request|skills\.runtime" || status=1
require_match "Concrete runtime skill implementation" "services/orion-cortex-exec/app/verb_adapters.py" "@verb\(\"skills\.runtime\.docker_prune_stopped_containers\.v1\"\)" || status=1

echo "=== Most likely live trigger command ==="
cat <<'CMD'
curl -sS -X POST "http://127.0.0.1:${HUB_PORT:-8000}/api/chat" \
  -H "Content-Type: application/json" \
  --data '{
    "mode": "agent",
    "messages": [{"role": "user", "content": "Dry-run cleanup of stopped containers."}],
    "use_recall": false,
    "no_write": true
  }'
CMD

echo
if [[ $status -ne 0 ]]; then
  echo "PATH_PROOF=FAIL"
  exit 1
fi

echo "PATH_PROOF=PASS"
