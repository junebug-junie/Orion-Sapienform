#!/usr/bin/env bash
set -euo pipefail
command -v docker >/dev/null || { echo "❌ Docker not found. Run base bootstrap first."; exit 1; }
mkdir -p /mnt/data/models /mnt/data/voice-cache
if [[ ! -d /mnt/data/docker/orion-docker-stack ]]; then
  echo "❌ Stack repo missing. Ensure agent forwarding works and rerun bootstrap."
  exit 1
fi
cd /mnt/data/docker/orion-docker-stack
docker compose -f docker-compose.voice.yml up -d || true
IP=$(hostname -I | awk '{print $1}')
echo "✅ Voice stack attempted. http://$IP:8000"
