#!/usr/bin/env bash
set -euo pipefail

# Validates a built orion-llamacpp-host image for:
# 1) pinned build tag/version visibility
# 2) CUDA availability
# 3) Qwen2.5 load + inference
# 4) Qwen3 load + inference
# 5) llama-server startup

IMAGE="${IMAGE:-orion-llamacpp-host:0.1.0}"
MODEL_QWEN25_PATH="${MODEL_QWEN25_PATH:-}"
MODEL_QWEN3_PATH="${MODEL_QWEN3_PATH:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${MODEL_QWEN25_PATH}" || -z "${MODEL_QWEN3_PATH}" ]]; then
  echo "ERROR: set MODEL_QWEN25_PATH and MODEL_QWEN3_PATH to concrete GGUF files on the host." >&2
  exit 1
fi

if [[ ! -f "${MODEL_QWEN25_PATH}" ]]; then
  echo "ERROR: MODEL_QWEN25_PATH not found: ${MODEL_QWEN25_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_QWEN3_PATH}" ]]; then
  echo "ERROR: MODEL_QWEN3_PATH not found: ${MODEL_QWEN3_PATH}" >&2
  exit 1
fi

echo "[1/7] version check"
docker run --rm --entrypoint /bin/bash "${IMAGE}" -lc 'set -e; /app/llama-server --version; /app/llama-cli --version'

echo "[2/7] CUDA visibility check"
docker run --rm --gpus all --entrypoint /bin/bash "${IMAGE}" -lc 'set -e; nvidia-smi -L'

run_infer() {
  local host_model="$1"
  local name="$2"
  local prompt="$3"

  local model_dir
  local model_file
  model_dir="$(dirname "${host_model}")"
  model_file="$(basename "${host_model}")"

  echo "[${name}] quick llama-cli inference"
  docker run --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -v "${model_dir}:/models-ro:ro" \
    --entrypoint /bin/bash "${IMAGE}" -lc "\
      set -euo pipefail; \
      /app/llama-cli -m /models-ro/${model_file} --n-gpu-layers 99 --ctx-size 1024 -n 24 -p '${prompt}' 2>&1 | tee /tmp/${name}.log; \
      grep -Eiq 'CUDA|ggml_cuda|offload|GPU' /tmp/${name}.log"
}

run_server_boot() {
  local host_model="$1"
  local name="$2"

  local model_dir
  local model_file
  model_dir="$(dirname "${host_model}")"
  model_file="$(basename "${host_model}")"

  echo "[${name}] llama-server startup"
  docker run --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -p 18080:8080 \
    -v "${model_dir}:/models-ro:ro" \
    --entrypoint /bin/bash "${IMAGE}" -lc "\
      set -euo pipefail; \
      /app/llama-server -m /models-ro/${model_file} --host 0.0.0.0 --port 8080 --n-gpu-layers 99 --ctx-size 1024 > /tmp/${name}-server.log 2>&1 & \
      pid=\$!; \
      for _ in \$(seq 1 30); do \
        if curl -fsS http://127.0.0.1:8080/health >/dev/null; then break; fi; \
        sleep 1; \
      done; \
      curl -fsS http://127.0.0.1:8080/health; \
      kill \$pid; wait \$pid || true"
}

echo "[3/7] existing Qwen2/Qwen2.5 model load"
run_infer "${MODEL_QWEN25_PATH}" "qwen25" "Reply with exactly: qwen25-ok"

echo "[4/7] Qwen3 model load"
run_infer "${MODEL_QWEN3_PATH}" "qwen3" "Reply with exactly: qwen3-ok"

echo "[5/7] llama-server startup (Qwen2.5)"
run_server_boot "${MODEL_QWEN25_PATH}" "qwen25"

echo "[6/7] llama-server startup (Qwen3)"
run_server_boot "${MODEL_QWEN3_PATH}" "qwen3"

echo "[7/7] Basic inference checks complete for both model families."
echo "Done. Review output for any changed warnings/flags." 
