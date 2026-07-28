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

## Post-bootstrap: host crontab jobs (do not skip this)
None of the steps above touch the host crontab. This repo has several host-level jobs (disk/DB maintenance, health watchdogs) that run via plain `crontab`, not `docker-compose` and not this bootstrap script -- they do **not** come back automatically on a new machine or after a host rebuild. If this box is ever replaced or rebuilt, these must be reinstalled by hand.

`scripts/README.md` is the authoritative, current list of every host-cron script this repo ships and its exact install command (search that file for "cron" or `crontab -e`) -- check there for the full and up-to-date set, since new ones get added over time and duplicating the list here would just rot. As of this writing that includes at minimum:
- Fuseki recover/compact (`services/orion-rdf-store/README.md`)
- Concept-relation digest (`services/orion-memory-consolidation/README.md`)
- Bus-core crash-loop watchdog (`scripts/bus_core_health_watchdog.py`, see `scripts/README.md`)
- Disk threshold watchdog (`scripts/disk_threshold_watchdog.py`, see `scripts/README.md`)

Exact contents of the currently-live crontab on the existing host (`crontab -l`, verified 2026-07-28), as a copy-paste starting point -- **do not assume this list is complete**, cross-check against `scripts/README.md` for anything shipped since:
```cron
5,25,45 * * * * cd /mnt/scripts/Orion-Sapienform/services/orion-rdf-store && make recover >> /mnt/scripts/Orion-Sapienform/logs/orion-fuseki-recover.log 2>&1
0 12 * * * cd /mnt/scripts/Orion-Sapienform/services/orion-rdf-store && SOURCE=/mnt/graphdb/rdf-store/fuseki/databases/orion make compact >> /mnt/graphdb/rdf_logs/fuseki-compact-run.log 2>&1
*/30 * * * * PATH=/mnt/scripts/Orion-Sapienform/venv/bin:$PATH POSTGRES_URI=$(grep -m1 '^POSTGRES_URI=' /mnt/scripts/Orion-Sapienform/services/orion-hub/.env | cut -d= -f2-) make -C /mnt/scripts/Orion-Sapienform concept-relation-digest >> /mnt/scripts/Orion-Sapienform/logs/orion-concept-relation-digest.log 2>&1
```
Note the bus-core and disk-threshold watchdogs are documented in `scripts/README.md` with their own install lines but are **not** in the block above -- confirm against a fresh `crontab -l` on the actual current host before treating this snippet as ground truth, don't just copy it blind.
