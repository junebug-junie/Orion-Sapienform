#!/usr/bin/env bash
set -euo pipefail
echo "🔍 Host nvidia-smi:"
command -v nvidia-smi >/dev/null || { echo "❌ nvidia-smi not found. Reboot or install driver."; exit 1; }
nvidia-smi || true
echo
echo "🔍 Docker GPU:"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi || { echo "❌ GPU not visible in Docker."; exit 1; }
echo "✅ GPU OK inside Docker."
