#!/usr/bin/env bash
# Live Gates A/B: Qwen3 with thinking disabled (argv + per-request kwargs).
# Requires Docker, GPU (--gpus all), jq, curl. MODEL_QWEN3_PATH must point to a Qwen3 GGUF on the host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

model_dir="$(dirname "${MODEL_QWEN3_PATH}")"
model_file="$(basename "${MODEL_QWEN3_PATH}")"

echo "Gate A: POST /apply-template (PR #13196-style body, enable_thinking false)"
echo "Gate B: POST /v1/chat/completions (marker + forbidden substrings from goldens/)"
echo "IMAGE=${IMAGE} PORT=${PORT} MODEL_FILE=${model_file}"

docker run --rm --gpus all \
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
