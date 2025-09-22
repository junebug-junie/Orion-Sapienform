#!/usr/bin/env bash
set -euo pipefail

LOGFILE="/var/log/orion-bootstrap.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "🚀 Orion Mesh Bootstrap (Required Tailscale + Agent Forwarding)"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

apt update && apt -y upgrade

bash "$BASE_DIR/install-utils.sh"
bash "$BASE_DIR/setup-node.sh"
bash "$BASE_DIR/setup-ssh.sh"
bash "$BASE_DIR/setup-rsfp-10g.sh"
bash "$BASE_DIR/install-docker.sh"

# ===== REQUIRED: Tailscale =====
AUTH_KEY="${ORION_TAILSCALE_AUTH_KEY:-}"
if [[ -z "$AUTH_KEY" && -s "$BASE_DIR/../config/tailscale-auth-key.txt" ]]; then
  AUTH_KEY="$(tr -d '\n' < "$BASE_DIR/../config/tailscale-auth-key.txt")"
fi
if [[ -z "$AUTH_KEY" ]]; then
  echo "❌ Tailscale auth key is required. Provide ORION_TAILSCALE_AUTH_KEY or config/tailscale-auth-key.txt"
  exit 1
fi
echo "🌐 Installing and bringing up Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh
# Try to start tailscaled if not running
systemctl enable --now tailscaled 2>/dev/null || true
tailscale up --auth-key "$AUTH_KEY" || { echo "❌ tailscale up failed. Check key and network."; exit 1; }
tailscale status || true

# ===== Optional GPU stack =====
if [[ "${ORION_INSTALL_NVIDIA:-1}" == "1" ]]; then
  bash "$BASE_DIR/install-nvidia.sh"
else
  echo "⏭️  Skipping NVIDIA install (ORION_INSTALL_NVIDIA=0)."
fi

# ===== Git clone via agent forwarding =====
export GIT_SSH_COMMAND="ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

mkdir -p /mnt/data/docker
REPO_URL="$(tr -d '\n' < "$BASE_DIR/../config/docker-compose.repo" || true)"
if [[ -z "$REPO_URL" ]]; then
  echo "❌ config/docker-compose.repo is empty. Set SSH URL (git@github.com:ORG/REPO.git)."
else
  if [[ ! -d /mnt/data/docker/orion-docker-stack ]]; then
    echo "🔗 Cloning $REPO_URL"
    if git ls-remote "$REPO_URL" >/dev/null 2>&1; then
      git clone "$REPO_URL" /mnt/data/docker/orion-docker-stack
    else
      echo "❌ SSH/agent not authorized. Run scripts/verify-agent.sh, then re-run."
      exit 1
    fi
  else
    echo "🔄 Updating repo..."
    (cd /mnt/data/docker/orion-docker-stack && git pull || true)
  fi
fi

echo "✅ Bootstrap complete. Reboot recommended if driver installed."
