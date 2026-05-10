#!/usr/bin/env bash
# Live check: Atlas quick-lane (or any) llama-server — chat completion with per-request
# chat_template_kwargs.enable_thinking=false must not leak thinking delimiter substrings.
#
# Requires: curl, jq. No Docker.
#
#   export ATLAS_LLAMACPP_QUICK_URL=http://<tailscale-or-host>:8013   # no trailing slash
#   export ATLAS_QUICK_CHAT_MODEL=Qwen_Qwen3-8B-Q4_K_M.gguf         # must match server model id
#   bash services/orion-llamacpp-host/scripts/verify_atlas_quick_llamacpp_thinking_off.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${ATLAS_LLAMACPP_QUICK_URL:?ERROR: set ATLAS_LLAMACPP_QUICK_URL (e.g. http://100.x.x.x:8013)}"
MODEL="${ATLAS_QUICK_CHAT_MODEL:?ERROR: set ATLAS_QUICK_CHAT_MODEL (GGUF filename as known to llama-server)}"

BASE="${BASE%/}"
MARKER="LIVE-ATLAS-QUICK-NO-THINK-OK"

echo "POST ${BASE}/v1/chat/completions model=${MODEL}"

resp="$(jq -n \
  --arg model "$MODEL" \
  --arg marker "$MARKER" \
  '{
    model: $model,
    messages: [{role: "user", content: ("Reply with exactly: " + $marker)}],
    max_tokens: 96,
    temperature: 0.2,
    chat_template_kwargs: {enable_thinking: false}
  }' | curl -fsS -X POST "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @-)"

content="$(printf '%s' "$resp" | jq -r '.choices[0].message.content // empty')"
if [[ "$content" != *"$MARKER"* ]]; then
  echo "ERROR: expected marker in assistant content" >&2
  printf '%s\n' "$resp" >&2
  exit 1
fi

while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" || "${line}" == \#* ]] && continue
  if printf '%s' "$content" | grep -Fq -- "$line"; then
    echo "ERROR: forbidden thinking marker in content: ${line}" >&2
    printf '%s\n' "$resp" >&2
    exit 1
  fi
done < "${SCRIPT_DIR}/goldens/forbidden_thinking_markers.txt"

echo "verify_atlas_quick_llamacpp_thinking_off.sh: OK"
