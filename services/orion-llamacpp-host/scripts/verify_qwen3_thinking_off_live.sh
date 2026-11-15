#!/usr/bin/env bash
# Live Gates A/B: Qwen3 with thinking disabled (argv + per-request kwargs).
# Requires Docker, GPU (--gpus all), jq, curl. MODEL_QWEN3_PATH must point to a Qwen3 GGUF on the host.
#
# This starts its OWN throwaway llama-server on one card and talks only to it (inside the container);
# it never calls a pool seat. But it uses a card, so it runs under an operator hold on the lane that
# owns that card (leased-evals design, docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md):
# `python -m orion.evals.leased_llm hold` waits behind Orion's own work, and when the pool takes
# the seat back it stops the container and exits 3. CUDA_VISIBLE_DEVICES picks the card (PCI order, as
# the pool numbers cards); HOLD_LANE defaults to circe's lane on it (0 chat, 1 agent, 2 agent-gpu2,
# 3 fast), so with the default it runs on circe only. Needs ORION_BUS_URL and the repo's python
# (ORION_PYTHON, default .venv/bin/python, else python3).
# What the hold does and does not buy: Orion's work waits (or recalls the check) instead of sharing the
# card unseen, and the run shows in gpu_pool_events. The lane's resident model stays loaded, so the
# throwaway server shares the card's VRAM with it -- pick a small GGUF, or reserve the whole card the way
# scripts/gpu_pool_clear_gpu2.py does for the power bench.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE="${IMAGE:-orion-llamacpp-host:0.1.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-18081}"
MODEL_QWEN3_PATH="${MODEL_QWEN3_PATH:-}"

if [[ -z "${MODEL_QWEN3_PATH}" ]]; then
  echo "ERROR: MODEL_QWEN3_PATH must be set to an absolute path to a Qwen3 GGUF on the host." >&2
  exit 1
fi
if [[ ! -f "${MODEL_QWEN3_PATH}" ]]; then
  echo "ERROR: MODEL_QWEN3_PATH not found: ${MODEL_QWEN3_PATH}" >&2
  exit 1
fi

: "${ORION_BUS_URL:?ERROR: set ORION_BUS_URL (the GPU pool hands out the operator hold over the bus)}"
case "${CUDA_VISIBLE_DEVICES}" in
  0) default_lane=chat ;; 1) default_lane=agent ;; 2) default_lane=agent-gpu2 ;; 3) default_lane=fast ;;
  *) default_lane="" ;;
esac
if [[ -z "${HOLD_LANE:-}" && "$(hostname -s)" != circe ]]; then
  echo "ERROR: the card-to-lane map is circe's; on $(hostname -s) set HOLD_LANE to the lane that owns the card" >&2
  exit 1
fi
HOLD_LANE="${HOLD_LANE:-${default_lane}}"
if [[ -z "${HOLD_LANE}" ]]; then
  echo "ERROR: set HOLD_LANE: no default lane for CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
PY="${ORION_PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${REPO}/.venv/bin/python" ]]; then PY="${REPO}/.venv/bin/python"; else PY=python3; fi
fi

model_dir="$(dirname "${MODEL_QWEN3_PATH}")"
model_file="$(basename "${MODEL_QWEN3_PATH}")"

echo "Gate A: POST /apply-template (PR #13196-style body, enable_thinking false)"
echo "Gate B: POST /v1/chat/completions (marker + forbidden substrings from goldens/)"
echo "IMAGE=${IMAGE} PORT=${PORT} MODEL_FILE=${model_file} HOLD_LANE=${HOLD_LANE}"

# A named container with --init: a recall's SIGTERM reaches llama-server (bash as PID 1 would ignore
# it), and the EXIT trap removes the container even if the docker CLI itself was killed.
NAME="qwen3-thinking-off-$$"
trap 'docker rm -f "${NAME}" >/dev/null 2>&1 || true' EXIT

PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m orion.evals.leased_llm hold \
  --lane "${HOLD_LANE}" --holder operator:verify-qwen3-thinking-off -- \
docker run --rm --init --name "${NAME}" --gpus all \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -p "${PORT}:8080" \
  -v "${model_dir}:/models-ro:ro" \
  -v "${SCRIPT_DIR}/goldens:/goldens-ro:ro" \
  --entrypoint /bin/bash "${IMAGE}" -lc "\
set -euo pipefail; \
/app/llama-server -m /models-ro/${model_file} --host 0.0.0.0 --port 8080 --n-gpu-layers 99 --ctx-size 1024 \
  --jinja --reasoning-budget 0 --chat-template-kwargs '{\"enable_thinking\":false}' \
  > /tmp/qwen3-thinking-off-server.log 2>&1 & \
pid=\$!; \
for _ in \$(seq 1 60); do \
  if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi; \
  sleep 1; \
done; \
curl -fsS http://127.0.0.1:8080/health >/dev/null; \
jq -n --arg model \"${model_file}\" --arg msg 'Give me a short introduction to large language models.' \
'{model: \$model, messages: [{role: \"user\", content: \$msg}], temperature: 0.7, top_p: 0.8, top_k: 20, max_tokens: 8192, presence_penalty: 1.5, chat_template_kwargs: {enable_thinking: false}}' \
| curl -fsS -X POST http://127.0.0.1:8080/apply-template -H 'Content-Type: application/json' -d @- \
| jq -e '.prompt != null and (.prompt|length) > 20' >/dev/null; \
resp=\$(jq -n --arg model \"${model_file}\" \
'{model: \$model, messages: [{role: \"user\", content: \"Reply with exactly: LIVE-GATE-B-OK\"}], max_tokens: 64, temperature: 0.2, chat_template_kwargs: {enable_thinking: false}}' \
| curl -fsS -X POST http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d @-); \
content=\$(printf '%s' \"\${resp}\" | jq -r '.choices[0].message.content // empty'); \
if [[ \"\${content}\" != *\"LIVE-GATE-B-OK\"* ]]; then echo 'Gate B: expected LIVE-GATE-B-OK in assistant content' >&2; printf '%s\n' \"\${resp}\" >&2; kill \${pid} 2>/dev/null || true; wait \${pid} 2>/dev/null || true; exit 1; fi; \
while IFS= read -r line || [[ -n \"\${line}\" ]]; do \
  [[ -z \"\${line}\" || \"\${line}\" == \\#* ]] && continue; \
  if printf '%s' \"\${content}\" | grep -Fq -- \"\${line}\"; then echo \"Gate B: forbidden thinking marker in content: \${line}\" >&2; printf '%s\n' \"\${resp}\" >&2; kill \${pid} 2>/dev/null || true; wait \${pid} 2>/dev/null || true; exit 1; fi; \
done < /goldens-ro/forbidden_thinking_markers.txt; \
kill \${pid}; wait \${pid} || true"

echo "verify_qwen3_thinking_off_live.sh: OK"
