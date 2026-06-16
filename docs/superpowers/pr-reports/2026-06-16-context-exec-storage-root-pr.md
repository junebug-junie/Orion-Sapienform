# PR Report: context-exec storage root + health (PR 1)

**Branch:** `feat/context-exec-storage-root`  
**Base:** `main`  
**Scope:** Mount durable NVMe-backed storage into context-exec, expose storage health on `/health`, and default proposal ledger paths away from `/tmp`.

## Problem

Context-exec had no durable storage contract. Docker compose only mounted the monorepo read-only at `/repo`, settings had no storage-root fields, and the proposal review ledger defaulted to `/tmp/orion-proposals.json` — ephemeral and invisible to operators.

## Solution

1. **Host mount layer only** — compose binds `${CONTEXT_EXEC_HOST_STORAGE_ROOT:-/mnt/rlm-nvme/context-exec}` → `/var/lib/orion/context-exec`. The app never references `/dev/nvme0n1` or host paths.
2. **Settings** — eight container-side path fields plus `CONTEXT_EXEC_RUN_LEDGER_ENABLED`.
3. **`app/storage.py`** — `ensure_storage_dirs()` on startup; `storage_health_block()` for `/health`.
4. **Compose + `.env_example`** — storage env block; proposal ledger defaults to `/var/lib/orion/context-exec/ledger/orion-proposals.json`.
5. **Tests** — extend `test_health.py` to assert storage block shape.
6. **README** — document storage arena and update ledger path examples.

## Host prep (operator, requires sudo)

```bash
lsblk -o NAME,MODEL,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS /dev/nvme0n1
sudo wipefs -n /dev/nvme0n1
# Only if disposable:
sudo parted /dev/nvme0n1 --script mklabel gpt mkpart primary 0% 100%
sudo mkfs.btrfs -f -L orion-rlm-nvme /dev/nvme0n1p1
sudo mkdir -p /mnt/rlm-nvme
sudo mount /dev/disk/by-label/orion-rlm-nvme /mnt/rlm-nvme
sudo mkdir -p /mnt/rlm-nvme/context-exec/{runs,artifacts,ledger,workspaces,cache,tmp,smoke-logs,locks}
# Persist in /etc/fstab, remount, migrate ledger if present
sudo cp -av /tmp/orion-proposals.json /mnt/rlm-nvme/context-exec/ledger/orion-proposals.json 2>/dev/null || true
```

## Files changed

- `services/orion-context-exec/docker-compose.yml`
- `services/orion-context-exec/app/settings.py`
- `services/orion-context-exec/app/storage.py` (NEW)
- `services/orion-context-exec/app/main.py`
- `services/orion-context-exec/app/api.py`
- `services/orion-context-exec/.env_example`
- `services/orion-context-exec/tests/test_health.py`
- `services/orion-context-exec/README.md`

## Verification

```bash
cd .worktrees/feat/context-exec-storage-root
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m pytest \
  services/orion-context-exec/tests/test_health.py -q --tb=short
# exit 0 — 3 passed

/mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m compileall \
  services/orion-context-exec/app/storage.py
# exit 0
```

Docker smoke (after host mount exists):

```bash
cd services/orion-context-exec
cp .env_example .env
docker compose up -d --build context-exec
curl -s http://localhost:8096/health | jq '.storage'
```

Expected: `configured: true`, `ok: true`, `root: "/var/lib/orion/context-exec"`, all subdirs writable.

Local `.env` sync (gitignored, not in PR): main checkout `services/orion-context-exec/.env` updated with storage keys from `.env_example`.

## Non-goals (this PR)

- Run persistence / workspace manager
- Bus channel or schema registry changes
- Behavior change beyond durable mount + health visibility + ledger path default

## Remaining risks

- Host NVMe prep requires sudo; not executed in agent session (permission denied on `/mnt/rlm-nvme`).
- Docker smoke against live mount UNVERIFIED until operator completes host prep.
- Code reviewer subagent hit API limit; manual self-review applied (README `/tmp` references updated).

## Code review (manual)

| Area | Result |
|------|--------|
| Spec compliance | All 7 patches implemented; no run persistence bleed |
| Host vs container paths | NVMe only in compose volume source; app uses `/var/lib/orion/context-exec` |
| Tests | Storage block shape covered; `ok` not asserted in unit tests (paths may be absent outside container) |
| Docs | README aligned with new defaults |
