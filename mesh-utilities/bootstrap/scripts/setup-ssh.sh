#!/usr/bin/env bash
set -euo pipefail
apt install -y openssh-server
systemctl enable --now ssh

# Inbound admin keys (public only)
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [[ -s "$(dirname "$0")/../keys/admin_authorized_keys.pub" ]]; then
  cat "$(dirname "$0")/../keys/admin_authorized_keys.pub" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  echo "🔑 Installed admin public keys to /root/.ssh/authorized_keys"
else
  echo "ℹ️ No admin_authorized_keys.pub found; inbound SSH remains password-based unless disabled."
fi

# Pre-seed GitHub host key for outbound convenience (agent forwarding required)
if ! grep -q "github.com" /root/.ssh/known_hosts 2>/dev/null; then
  ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null || true
fi
