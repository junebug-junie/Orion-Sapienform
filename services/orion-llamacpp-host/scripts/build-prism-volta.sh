#!/bin/bash
# Build orion-llamacpp-host-prism:0.1.0 on circe: the stock llamacpp-host image plus PrismML's
# llama.cpp fork at /app/prism (Dockerfile.prism, GPU pool stage 7.2). atlas-agent-burst (pool
# role agent-gpu2) runs this image; the lane controller starts it with `up --no-build`, so build
# it BEFORE the pool's next load of agent-gpu2. Building does not touch any running container.
#
# Run from a worktree at the deployed commit, repo root:
#   services/orion-llamacpp-host/scripts/build-prism-volta.sh
# LLAMACPP_IMAGE_TAG (the stock half) is read from services/orion-llamacpp-host/.env, the same
# value every other atlas worker is built with.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

IMAGE="${PRISM_LLAMACPP_IMAGE:-orion-llamacpp-host-prism:0.1.0}"
# The stock half must match the other atlas workers (the Q4 rollback runs on it). A worktree has
# no .env (gitignored), so fall back to the primary checkout's; never guess a default tag.
PRIMARY="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
TAG="${LLAMACPP_IMAGE_TAG:-}"
for ENV_FILE in "services/orion-llamacpp-host/.env" "${PRIMARY}/services/orion-llamacpp-host/.env"; do
  if [ -z "${TAG}" ] && [ -f "${ENV_FILE}" ]; then
    TAG="$(grep -E '^LLAMACPP_IMAGE_TAG=' "${ENV_FILE}" | tail -n1 | cut -d= -f2- || true)"
  fi
done
if [ -z "${TAG}" ]; then
  echo "LLAMACPP_IMAGE_TAG not found in services/orion-llamacpp-host/.env; set it (same tag as the other atlas workers)" >&2
  exit 1
fi

echo "building ${IMAGE} (stock half: ghcr.io/ggml-org/llama.cpp:${TAG}; fork: Prism b10750, sm_70 / CUDA 12.8)"
docker build \
  -f services/orion-llamacpp-host/Dockerfile.prism \
  --build-arg LLAMACPP_IMAGE_TAG="${TAG}" \
  -t "${IMAGE}" \
  .

echo "checking both binaries run in the final image"
docker run --rm --gpus all --entrypoint /bin/sh "${IMAGE}" -c \
  'LD_LIBRARY_PATH=/app/prism${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} /app/prism/llama-server --version && /app/prism/llama-server --help >/dev/null'
docker run --rm --gpus all --entrypoint /app/llama-server "${IMAGE}" --version
echo "ok ${IMAGE}"
