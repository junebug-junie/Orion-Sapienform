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
# TWO-PASS CHECK/REPAIR (verified by local reproduction against a real
# healthy AOF and a real mid-file-corrupted AOF, 2026-07-16):
# `redis-check-aof` (no --fix) is non-destructive, non-interactive, and
# near-instant: on a healthy AOF it prints "All AOF files and manifest are
# valid" and exits 0; on a corrupted AOF it prints "... is not valid. Use
# the --fix option to try fixing it." and exits 1 -- WITHOUT ever touching
# stdin or modifying any file. We run this plain check first and only
# escalate to the destructive --fix path when it reports a problem (exit
# != 0). This means a healthy boot -- the overwhelming common case --
# never runs the destructive path at all.
#
# IMPORTANT -- confirmed by local reproduction, not assumed:
# `redis-check-aof --fix` is interactive *whenever it actually finds
# something to repair*: it prints "Continue? [y/N]:" and reads stdin.
# There is no --yes/--force/non-interactive flag (checked via `strings` on
# the redis:7-alpine binary). Feeding it a closed stdin (e.g. `</dev/null`,
# or letting Docker give it no tty/pipe) does NOT hang -- it reads EOF,
# defaults to "N", and ABORTS WITHOUT FIXING (exit 1), leaving the AOF
# exactly as broken as before. That is a silent-looking failure mode we
# must not ship. So we explicitly feed it confirmation via `yes` piped
# (not a single `printf 'y\n'`): today's AOF auto-GC keeps the manifest to
# one base+incr file, so --fix only ever prints one prompt, but `yes`
# supplies "y" to every prompt until the pipe closes, so this stays correct
# even if AOF retention config changes and a repair spans multiple
# incr files needing multiple confirmations. Verified empirically that
# feeding `yes` more input than --fix reads causes no different behavior
# than a single `printf 'y\n'` (the unread lines are simply never consumed;
# --fix exits 0 and reports "All AOF files and manifest are valid" the same
# way either way).
#
# `--fix` is destructive by design: it TRUNCATES the AOF at the first
# corrupted record it finds, discarding everything after that point --
# confirmed empirically that a well-formed record placed after the
# corruption point is discarded too, not just a torn final write. Because
# of that, we take a plain-copy backup of whatever `--fix` is about to
# operate on (the whole appendonlydir, or the legacy single AOF file)
# before invoking it, so a human can recover the pre-repair bytes if the
# automatic repair discards more than expected. This is a single point-in-
# time copy, not a rotation/retention scheme -- an operator cleans up old
# backups manually.
#
# On an already-healthy AOF, we never reach --fix at all (see two-pass note
# above), so there is no backup, no prompt, and no modification -- confirmed
# idempotent and cheap: the non-destructive check of a small healthy AOF
# completes in well under a second, and cost scales with AOF size the same
# way Redis's own AOF-load scan already does on every boot, so this does
# not add a new order of magnitude of startup cost.
#
# If the repair itself fails (corruption redis-check-aof cannot resolve),
# fail loudly and refuse to start redis-server, rather than silently
# starting on an unknown/lossy state or silently skipping the check.

set -e

DATA_DIR="${REDIS_DATA_DIR:-/data}"
AOF_DIR="$DATA_DIR/appendonlydir"
MANIFEST="$AOF_DIR/appendonly.aof.manifest"
LEGACY_AOF="$DATA_DIR/appendonly.aof"
BACKUP_ROOT="$DATA_DIR/aof-repair-backups"

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

  echo "[entrypoint] Checking AOF at $repair_target before starting redis-server (non-destructive pass) ..."
  if redis-check-aof "$repair_target"; then
    echo "[entrypoint] AOF check passed -- healthy, no repair needed. Skipping destructive --fix pass entirely."
  else
    echo "[entrypoint] AOF check reported a problem -- escalating to redis-check-aof --fix."

    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    backup_dir="$BACKUP_ROOT/$timestamp"
    mkdir -p "$backup_dir"
    if [ "$repair_target" = "$MANIFEST" ]; then
      cp -a "$AOF_DIR" "$backup_dir/appendonlydir"
    else
      cp -a "$LEGACY_AOF" "$backup_dir/appendonly.aof"
    fi
    echo "[entrypoint] Backed up pre-repair AOF state to $backup_dir before running --fix -- inspect this path if the repair discards more than expected."

    if yes | redis-check-aof --fix "$repair_target"; then
      echo "[entrypoint] AOF check/repair completed -- proceeding to start redis-server."
    else
      echo "[entrypoint] FATAL: redis-check-aof --fix failed against $repair_target (exit $?)." >&2
      echo "[entrypoint] The AOF may be unrecoverable by automatic repair (e.g. the manifest references a missing/partial base or incr file). Refusing to start redis-server rather than risk starting on unknown/corrupt data." >&2
      echo "[entrypoint] Pre-repair state was backed up to $backup_dir. Manual intervention required -- inspect $DATA_DIR before restarting bus-core." >&2
      exit 1
    fi
  fi
else
  echo "[entrypoint] No AOF file found under $DATA_DIR (first boot, fresh volume, or AOF disabled) -- skipping repair step."
fi

echo "[entrypoint] Starting: $*"
exec "$@"
