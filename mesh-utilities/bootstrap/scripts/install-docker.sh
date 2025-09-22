#!/usr/bin/env bash
set -euo pipefail
curl -fsSL https://get.docker.com | sh
if [[ -n "${SUDO_USER:-}" ]]; then
  usermod -aG docker "$SUDO_USER" || true
fi
apt install -y docker-compose-plugin
apt install -y docker-compose
systemctl enable docker
systemctl restart docker
