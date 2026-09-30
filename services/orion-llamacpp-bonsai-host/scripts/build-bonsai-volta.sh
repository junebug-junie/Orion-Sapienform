#!/bin/bash
# Build the Circe Volta image for Ternary-Bonsai-2-27B (PrismML llama.cpp `prism` @ 88c4bc6).
# Host nvcc 13.x dropped sm_70 -- this uses CUDA 12.8 inside Docker.
#
# Run from a worktree, repo root:
#   services/orion-llamacpp-bonsai-host/scripts/build-bonsai-volta.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

IMAGE="${BONSAI_LLAMACPP_IMAGE:-llamacpp-bonsai-prism:server-local-volta}"
HOST_IMAGE="${HOST_IMAGE:-orion-llamacpp-host:0.1.0}"

if ! docker image inspect "${HOST_IMAGE}" >/dev/null 2>&1; then
  echo "missing ${HOST_IMAGE}; build orion-llamacpp-host first (scripts/safe_docker_build.sh orion-llamacpp-host build)" >&2
  exit 1
fi

echo "building ${IMAGE} from ${HOST_IMAGE} (sm_70 / CUDA 12.8)"
docker build \
  -f services/orion-llamacpp-bonsai-host/Dockerfile \
  --build-arg HOST_IMAGE="${HOST_IMAGE}" \
  -t "${IMAGE}" \
  .

echo "checking the final image runs the fork, not the base image's llama-server"
docker run --rm --gpus all --entrypoint /app/llama-server "${IMAGE}" --version
docker run --rm --gpus all --entrypoint /app/llama-server "${IMAGE}" --help | grep -E -- "--ctx-checkpoints"
echo "ok ${IMAGE}"
