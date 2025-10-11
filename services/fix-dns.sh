#!/bin/bash
set -e

echo "🔧 Applying DNS + NVIDIA runtime fixes for Docker globally..."

# 1. Ensure system resolv.conf uses real nameservers
sudo mv /etc/resolv.conf /etc/resolv.conf.systemd.bak 2>/dev/null || true
sudo tee /etc/resolv.conf >/dev/null <<'EOF'
nameserver 8.8.8.8
nameserver 1.1.1.1
EOF

# 2. Overwrite Docker daemon.json
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
  "dns": ["8.8.8.8", "1.1.1.1"],
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "args": []
    }
  },
  "features": { "buildkit": true }
}
EOF

# 3. Restart docker cleanly
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart docker

echo "✅ Docker restarted with global DNS + BuildKit + NVIDIA runtime."
docker info | grep -A2 "Runtimes"
