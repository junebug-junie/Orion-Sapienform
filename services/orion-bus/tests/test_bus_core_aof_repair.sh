#!/bin/sh
# test_bus_core_aof_repair.sh — end-to-end smoke test for the bus-core
# AOF auto-repair entrypoint (services/orion-bus/entrypoint.sh).
#
# This proves, against the ACTUAL built image (Dockerfile.bus-core), that:
#   1. A healthy AOF starts clean and is left byte-for-byte unmodified.
#   2. A corrupted AOF (mid-file corruption, not just a clean EOF
#      truncation -- the failure mode that Redis's built-in
#      aof-load-truncated tolerance does NOT cover) makes the container
#      exit non-zero / fail health when run with the OLD stock image and
#      no repair step.
#   3. The NEW image (with entrypoint.sh) repairs the same corrupted AOF
#      on boot and Redis starts clean, serving PONG and the data that
#      existed before the corruption point.
#
# Uses ONLY a throwaway directory under /tmp -- never touches the live
# /mnt/telemetry/orion-athena/bus/data volume.
#
# Usage: sh services/orion-bus/tests/test_bus_core_aof_repair.sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SERVICE_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
# Suffix the image tag and container names with $$ (this process's PID), not
# just $WORKDIR -- this repo routinely has multiple agent worktrees running
# concurrently against the same host Docker daemon. Fixed literal names
# would collide across concurrent test runs: a second invocation's `docker
# run --name "$C1"` would fail with "name already in use",
# or worse, one run's cleanup() trap would `docker rm -f` a container that
# belongs to a different concurrently-running invocation.
IMAGE="${BUS_CORE_TEST_IMAGE:-orion-bus-core-test:local}.$$"
C1="bus-core-aof-test-c1.$$"
C2="bus-core-aof-test-c2.$$"
C3="bus-core-aof-test-c3.$$"
WORKDIR="${TMPDIR:-/tmp}/bus-core-aof-repair-test.$$"
FAIL=0

BUS_CMD='redis-server --appendonly yes --maxmemory-policy allkeys-lru --client-output-buffer-limit pubsub 64mb 32mb 120'

echo "== Building $IMAGE from $SERVICE_DIR/Dockerfile.bus-core =="
docker build -f "$SERVICE_DIR/Dockerfile.bus-core" -t "$IMAGE" "$SERVICE_DIR" >/dev/null
echo "OK: image built"

cleanup() {
  _rc=$?
  docker rm -f "$C1" "$C2" "$C3" >/dev/null 2>&1 || true
  # Files under $WORKDIR were created by the container's root user, not the
  # host user running this script -- a plain `rm -rf` can't remove them.
  # Use a throwaway container (same uid namespace as the AOF files) to clean
  # up instead, best-effort, so a cleanup failure never masks the test's
  # real pass/fail exit code.
  docker run --rm -v "$WORKDIR:/cleanup" "$IMAGE" sh -c 'rm -rf /cleanup/*' >/dev/null 2>&1 || true
  rm -rf "$WORKDIR" >/dev/null 2>&1 || true
  # $IMAGE is now uniquely tagged per invocation (see above), so unlike a
  # fixed shared tag, leftover images from repeated/parallel runs would
  # actually accumulate on disk if not removed here.
  docker rmi -f "$IMAGE" >/dev/null 2>&1 || true
  exit "$_rc"
}
trap cleanup EXIT

echo "== bus-core AOF auto-repair smoke test =="
echo "Image under test: $IMAGE"
echo "Scratch dir: $WORKDIR"

mkdir -p "$WORKDIR/healthy/data" "$WORKDIR/corrupt/data"

# --- Step 1: produce a healthy multi-part AOF using the real image -------
echo
echo "-- Step 1: seed a healthy AOF --"
docker run -d --name "$C1" -v "$WORKDIR/healthy/data:/data" "$IMAGE" $BUS_CMD >/dev/null
sleep 2
docker exec "$C1" redis-cli set foo bar >/dev/null
docker exec "$C1" redis-cli set baz qux >/dev/null
sleep 1
docker stop "$C1" >/dev/null
docker rm "$C1" >/dev/null

if [ ! -f "$WORKDIR/healthy/data/appendonlydir/appendonly.aof.manifest" ]; then
  echo "FAIL: expected multi-part AOF manifest not found after seeding"
  exit 1
fi
echo "OK: multi-part AOF layout confirmed at appendonlydir/appendonly.aof.manifest"

BEFORE_SUM=$(sha1sum "$WORKDIR/healthy/data/appendonlydir/appendonly.aof.1.incr.aof" | awk '{print $1}')

# --- Step 2: healthy AOF should boot clean AND be left untouched ---------
echo
echo "-- Step 2: healthy AOF boots clean and is left untouched by repair step --"
docker run -d --name "$C2" -v "$WORKDIR/healthy/data:/data" "$IMAGE" $BUS_CMD >/dev/null
sleep 2
if ! docker exec "$C2" redis-cli ping | grep -q PONG; then
  echo "FAIL: healthy container did not respond PONG"
  FAIL=1
fi
if [ "$(docker exec "$C2" redis-cli get foo)" != "bar" ]; then
  echo "FAIL: healthy container lost pre-existing data (foo)"
  FAIL=1
fi
docker logs "$C2" 2>&1 | grep -q "No AOF file found" && {
  echo "FAIL: repair step did not detect the existing AOF (logic bug)"
  FAIL=1
}
docker stop "$C2" >/dev/null
docker rm "$C2" >/dev/null

AFTER_SUM=$(sha1sum "$WORKDIR/healthy/data/appendonlydir/appendonly.aof.1.incr.aof" | awk '{print $1}')
if [ "$BEFORE_SUM" != "$AFTER_SUM" ]; then
  echo "FAIL: repair step modified an already-healthy AOF (should be a no-op)"
  FAIL=1
else
  echo "OK: healthy AOF byte-for-byte unmodified by repair step (idempotent)"
fi

# --- Step 3: build a corrupted AOF (mid-file corruption, not EOF trunc) --
echo
echo "-- Step 3: corrupt a copy of the AOF (mid-file, not a clean EOF truncation) --"
cp -r "$WORKDIR/healthy/data/appendonlydir" "$WORKDIR/corrupt/data/appendonlydir"
INCR="$WORKDIR/corrupt/data/appendonlydir/appendonly.aof.1.incr.aof"

python3 - "$INCR" <<'PYEOF'
import sys
path = sys.argv[1]
data = open(path, "rb").read()
# Corrupt the bulk-length header of the LAST command's value so it no
# longer matches the actual byte count that follows -- and append a
# trailing valid-looking record after it, so this is unambiguously
# mid-file corruption (Redis's aof-load-truncated tolerance only covers a
# clean truncation at end-of-file; a length/data mismatch followed by more
# bytes triggers the fatal "Bad file format" path instead).
marker = b"$3\r\nqux\r\n"
assert marker in data, "fixture assumption changed, update this test"
corrupted = data.replace(marker, b"$9\r\nqux\r\n", 1)
corrupted += b"*3\r\n$3\r\nset\r\n$5\r\nthird\r\n$6\r\nfourth\r\n"
open(path, "wb").write(corrupted)
print(f"corrupted AOF written: {len(data)} -> {len(corrupted)} bytes")
PYEOF

# --- Step 4: prove this corruption actually breaks a plain redis-server --
echo
echo "-- Step 4: confirm plain redis-server (no repair) fails to start on this AOF --"
PLAIN_LOG=$(docker run --rm -v "$WORKDIR/corrupt/data:/data" redis:7-alpine timeout 5 $BUS_CMD 2>&1 || true)
if echo "$PLAIN_LOG" | grep -q "Bad file format reading the append only file"; then
  echo "OK: reproduced the real failure mode (fatal 'Bad file format' error)"
else
  echo "FAIL: did not reproduce a fatal AOF error against plain redis-server -- fixture is not representative"
  echo "$PLAIN_LOG"
  FAIL=1
fi

# --- Step 5: the NEW image repairs it and starts clean -------------------
echo
echo "-- Step 5: new image (with entrypoint.sh) repairs the corrupted AOF and starts clean --"
docker run -d --name "$C3" -v "$WORKDIR/corrupt/data:/data" "$IMAGE" $BUS_CMD >/dev/null
sleep 2
REPAIR_LOG=$(docker logs "$C3" 2>&1)
echo "$REPAIR_LOG" | grep -q "AOF check/repair completed" || {
  echo "FAIL: entrypoint did not report a completed repair"
  echo "$REPAIR_LOG"
  FAIL=1
}
if ! docker exec "$C3" redis-cli ping 2>/dev/null | grep -q PONG; then
  echo "FAIL: repaired container did not come up and respond PONG"
  echo "$REPAIR_LOG"
  FAIL=1
else
  echo "OK: repaired container is up and responds PONG"
fi
if [ "$(docker exec "$C3" redis-cli get foo 2>/dev/null)" = "bar" ]; then
  echo "OK: data written before the corruption point survived the repair (foo=bar)"
else
  echo "FAIL: data before the corruption point was lost -- repair over-truncated"
  FAIL=1
fi
docker stop "$C3" >/dev/null
docker rm "$C3" >/dev/null

# --- Step 6: unrecoverable AOF (manifest references a missing file) -----
# Distinct corruption shape from Step 3's mid-file byte corruption: this
# simulates a crash mid-manifest-rotation, where the manifest points at a
# base/incr file that never finished being written. redis-check-aof cannot
# repair a missing file -- entrypoint.sh must fail closed (non-zero exit,
# clear FATAL log) rather than silently starting redis-server or silently
# skipping the check.
echo
echo "-- Step 6: manifest referencing a missing file is refused, not silently started --"
mkdir -p "$WORKDIR/unrecoverable/data/appendonlydir"
cp "$WORKDIR/healthy/data/appendonlydir/appendonly.aof.manifest" \
   "$WORKDIR/unrecoverable/data/appendonlydir/appendonly.aof.manifest"
cp "$WORKDIR/healthy/data/appendonlydir/appendonly.aof.1.base.rdb" \
   "$WORKDIR/unrecoverable/data/appendonlydir/appendonly.aof.1.base.rdb"
# Deliberately omit appendonly.aof.1.incr.aof, which the manifest still
# references.
C4="bus-core-aof-test-c4.$$"
UNRECOVERABLE_LOG=$(docker run --rm --name "$C4" -v "$WORKDIR/unrecoverable/data:/data" "$IMAGE" $BUS_CMD 2>&1 || true)
if echo "$UNRECOVERABLE_LOG" | grep -q "\[entrypoint\] FATAL: redis-check-aof --fix failed"; then
  echo "OK: entrypoint refused to start and logged FATAL for an unrepairable manifest"
else
  echo "FAIL: entrypoint did not fail closed on a manifest referencing a missing file"
  echo "$UNRECOVERABLE_LOG"
  FAIL=1
fi
if echo "$UNRECOVERABLE_LOG" | grep -q "Ready to accept connections"; then
  echo "FAIL: redis-server started anyway despite the unrepairable AOF -- entrypoint did not fail closed"
  FAIL=1
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "== ALL CHECKS PASSED =="
  exit 0
else
  echo "== SOME CHECKS FAILED =="
  exit 1
fi
