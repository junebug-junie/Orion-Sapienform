#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${NODE_NAME:-}" ]]; then
  read -p "Enter hostname for this node: " NODE_NAME
fi
hostnamectl set-hostname "$NODE_NAME"
grep -q "127.0.1.1\s\+$NODE_NAME" /etc/hosts || echo "127.0.1.1 $NODE_NAME" >> /etc/hosts
mkdir -p /mnt/data /mnt/configs /mnt/scratch
chmod 777 /mnt/scratch
