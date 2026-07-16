#!/bin/sh
# entrypoint.sh — bus-core (Redis) boot wrapper
#
# On every boot, before starting redis-server, run redis-check-aof against
# whatever AOF file(s) actually exist under /data. This repairs a
# corrupted/partially-written AOF tail -- the common failure after an
# unclean shutdown (host reboot, OOM, disk pressure) -- so Redis can start
# instead of crash-looping on the exact same broken state forever
# (`restart: unless-stopped` alone never recovers from this).
#
# AOF layout (verified against redis:7-alpine / Redis 7.4.9, 2026-07-16):
# `--appendonly yes` with no other aof-* overrides (which is exactly what
# bus-core's command uses) produces Redis 7's multi-part AOF layout:
#   /data/appendonlydir/appendonly.aof.manifest       (index of base+incr files)
#   /data/appendonlydir/appendonly.aof.<N>.base.rdb
#   /data/appendonlydir/appendonly.aof.<N>.incr.aof
# There is no single top-level /data/appendonly.aof file in this layout.
# redis-check-aof takes the *manifest* (not the individual base/incr files)
# and repairs everything it references. We also check for the legacy
# single-file layout defensively, in case a data volume is ever migrated in
# from an older Redis version that used it.
#
# IMPORTANT -- confirmed by local reproduction, not assumed:
# `redis-check-aof --fix` is interactive *whenever it actually finds
# something to repair*: it prints "Continue? [y/N]:" and reads stdin.
# There is no --yes/--force/non-interactive flag (checked via `strings` on
# the redis:7-alpine binary). Feeding it a closed stdin (e.g. `</dev/null`,
# or letting Docker give it no tty/pipe) does NOT hang -- it reads EOF,
# defaults to "N", and ABORTS WITHOUT FIXING (exit 1), leaving the AOF
# exactly as broken as before. That is a silent-looking failure mode we
# must not ship. So we explicitly feed it "y".
#
# On an already-healthy AOF, --fix finds nothing to repair, never prints
# the confirmation prompt, reports "All AOF files and manifest are valid",
# and exits 0 without modifying the file at all -- confirmed idempotent and
# cheap: a full scan of a small healthy AOF completes in well under a
# second, and cost scales with AOF size the same way Redis's own AOF-load
# scan already does on every boot, so this does not add a new order of
# magnitude of startup cost.
#
# If the repair itself fails (corruption redis-check-aof cannot resolve),
# fail loudly and refuse to start redis-server, rather than silently
# starting on an unknown/lossy state or silently skipping the check.

set -e

DATA_DIR="${REDIS_DATA_DIR:-/data}"
AOF_DIR="$DATA_DIR/appendonlydir"
MANIFEST="$AOF_DIR/appendonly.aof.manifest"
LEGACY_AOF="$DATA_DIR/appendonly.aof"

repair_target=""
if [ -f "$MANIFEST" ]; then
  repair_target="$MANIFEST"
elif [ -f "$LEGACY_AOF" ]; then
  repair_target="$LEGACY_AOF"
fi

if [ -n "$repair_target" ]; then
  if ! command -v redis-check-aof >/dev/null 2>&1; then
    echo "[entrypoint] FATAL: redis-check-aof binary not found on PATH." >&2
    echo "[entrypoint] This means the base image no longer ships the tool this repair step depends on (e.g. a redis:7-alpine bump changed its layout) -- NOT that the AOF itself is corrupt." >&2
    echo "[entrypoint] Refusing to start redis-server unrepaired. Fix Dockerfile.bus-core / the base image before restarting bus-core." >&2
    exit 1
  fi
  echo "[entrypoint] Checking AOF at $repair_target before starting redis-server ..."
  if printf 'y\n' | redis-check-aof --fix "$repair_target"; then
    echo "[entrypoint] AOF check/repair completed -- proceeding to start redis-server."
  else
    echo "[entrypoint] FATAL: redis-check-aof --fix failed against $repair_target (exit $?)." >&2
    echo "[entrypoint] The AOF may be unrecoverable by automatic repair (e.g. the manifest references a missing/partial base or incr file). Refusing to start redis-server rather than risk starting on unknown/corrupt data." >&2
    echo "[entrypoint] Manual intervention required -- inspect $DATA_DIR before restarting bus-core." >&2
    exit 1
  fi
else
  echo "[entrypoint] No AOF file found under $DATA_DIR (first boot, fresh volume, or AOF disabled) -- skipping repair step."
fi

echo "[entrypoint] Starting: $*"
exec "$@"
