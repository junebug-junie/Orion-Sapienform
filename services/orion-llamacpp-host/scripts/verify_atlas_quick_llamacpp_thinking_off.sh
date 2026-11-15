#!/usr/bin/env bash
# Live check: the fast lane (or another operator lane) -- a chat completion with per-request
# chat_template_kwargs.enable_thinking=false must not leak thinking delimiter substrings.
#
# Leased (docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md): the call goes
# through orion-llm-gateway under an operator hold from the GPU pool (background priority, behind
# Orion's own work), via `python -m orion.evals.leased_llm chat`. No direct worker URL: the lane door
# refuses an unkeyed model call. Exit 3 = the pool took the seat back; 2 = no seat.
#
# Requires: jq, the repo's python (ORION_PYTHON, default .venv/bin/python, else python3), ORION_BUS_URL.
#
#   export ORION_BUS_URL=redis://100.92.216.81:6379/0
#   bash services/orion-llamacpp-host/scripts/verify_atlas_quick_llamacpp_thinking_off.sh   # LANE=fast
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LANE="${LANE:-fast}"
: "${ORION_BUS_URL:?ERROR: set ORION_BUS_URL (the GPU pool hands out the operator hold over the bus)}"
PY="${ORION_PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${REPO}/.venv/bin/python" ]]; then PY="${REPO}/.venv/bin/python"; else PY=python3; fi
fi
MARKER="LIVE-ATLAS-QUICK-NO-THINK-OK"

echo "lane=${LANE}: one leased call through the gateway"

set +e
resp="$(jq -n --arg marker "$MARKER" \
  '{
    messages: [{role: "user", content: ("Reply with exactly: " + $marker)}],
    max_tokens: 96,
    temperature: 0.2,
    chat_template_kwargs: {enable_thinking: false}
  }' | PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m orion.evals.leased_llm chat \
    --lane "${LANE}" --holder operator:verify-quick-thinking-off)"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "ERROR: leased call failed (exit ${rc}; 2 = no seat, 3 = recalled)" >&2
  printf '%s\n' "$resp" >&2
  exit "$rc"
fi

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
