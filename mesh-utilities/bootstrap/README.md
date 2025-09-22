# Orion Node Bootstrap (Ubuntu 24.04)

## What you MUST provide
1. **Tailscale auth key** (ephemeral or reusable) via either:
   - Env: `ORION_TAILSCALE_AUTH_KEY=<tskey>`
   - Or file: `orion-bootstrap/config/tailscale-auth-key.txt` (single line)
2. **Your admin SSH *public* key** in `orion-bootstrap/keys/admin_authorized_keys.pub` (optional but recommended).
   - If present, it will be appended to `/root/.ssh/authorized_keys` for inbound admin access.
3. **Your Docker stack repo SSH URL** in `orion-bootstrap/config/docker-compose.repo`:
   ```
   git@github.com:<username-or-org>/orion-docker-stack.git
   ```

> Outbound Git **never** uses on-disk keys in this build. It requires **agent forwarding** from your admin box (Carbon).

## Quick Start (Carbon → newmachine)
```bash
# On Carbon
ssh-add ~/.ssh/carbon-to-git-sh-ed25519
rsync -a ~/orion-bootstrap/ newmachine@<newmachine-IP>:~/orion-bootstrap-new/
ssh -tt newmachine@<newmachine-IP> 'sudo rm -rf /opt/orion-bootstrap && sudo mkdir -p /opt/orion-bootstrap && sudo rsync -a ~/orion-bootstrap-new/ /opt/orion-bootstrap && sudo chown -R root:root /opt/orion-bootstrap && sudo find /opt/orion-bootstrap -type f -name "*.sh" -exec sudo chmod +x {} \;'
ssh -A newmachine@<newmachine-IP>
```

On new machine:
```bash
# Required: set repo SSH URL
echo 'git@github.com:<username-or-org>/orion-docker-stack.git' | sudo tee /opt/orion-bootstrap/config/docker-compose.repo

# Provide Tailscale key via env OR file (env shown here)
ORION_TAILSCALE_AUTH_KEY=<tskey> sudo -E bash /opt/orion-bootstrap/scripts/orion-bootstrap.sh
# (skip GPU this pass) ORION_INSTALL_NVIDIA=0 ORION_TAILSCALE_AUTH_KEY=<tskey> sudo -E bash /opt/orion-bootstrap/scripts/orion-bootstrap.sh
```

Verify:
```bash
sudo bash /opt/orion-bootstrap/scripts/verify-agent.sh
sudo bash /opt/orion-bootstrap/scripts/verify-gpu.sh   # after driver install + reboot if needed
```

## Keys folder policy
- `keys/admin_authorized_keys.pub` — place one or more **public** keys (newline-separated). Appended to root's `authorized_keys`.
- No deploy/private keys are used or stored. All outbound Git uses the forwarded agent from Carbon.
