#!/bin/bash
# Build the Circe Volta image for DeepSeek-V4.1-Flash (dsv41-porte @ 3b6fcfe).
# Host nvcc 13.x dropped sm_70 — this uses CUDA 12.8 inside Docker.
#
# Run from a worktree, repo root:
#   services/orion-llamacpp-host/scripts/build-dsv41-volta.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

IMAGE="${DSV41_LLAMACPP_IMAGE:-llamacpp-dsv41-porte:server-local-volta}"
HOST_IMAGE="${HOST_IMAGE:-orion-llamacpp-host:0.1.0}"

echo "building ${IMAGE} from ${HOST_IMAGE} (sm_70 / CUDA 12.8)"
docker build \
  -f services/orion-llamacpp-host/Dockerfile.dsv41-porte \
  --build-arg HOST_IMAGE="${HOST_IMAGE}" \
  -t "${IMAGE}" \
  .

echo "checking --moe-stream on ${IMAGE}"
docker run --rm --entrypoint /app/llama-server "${IMAGE}" --help | grep -E -- "--moe-stream"
echo "ok ${IMAGE}"
