#!/usr/bin/env bash
# Regression test for the 2026-07-15 fix: fuseki_tdb_compact.sh and
# fuseki_recover.sh coordinate via flock on a shared lock file. This was
# previously a plain `[ -f lockfile ]` existence check, which was a real
# TOCTOU race that caused 4 consecutive weekly production compact failures
# (see README.md "Scheduled maintenance" section for the incident).
#
# This test doesn't touch Docker/Fuseki -- it validates the locking primitive
# itself in isolation: does a concurrent flock probe correctly detect
# contention while a holder is active, and correctly succeed once released?
set -euo pipefail

LOCK_FILE="$(mktemp -u /tmp/test-compact-lock-coordination.XXXXXX)"
trap 'rm -f "${LOCK_FILE}"' EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

# --- Test 1: a concurrent holder blocks a probing flock -n ---
(
  exec 200>"${LOCK_FILE}"
  flock -n 200 || exit 1
  sleep 2
) &
holder_pid=$!
sleep 0.5

(
  exec 9>"${LOCK_FILE}"
  if flock -n 9; then
    echo "prober acquired lock while holder was still active" >&2
    exit 1
  fi
  exit 0
) || fail "Test 1: concurrent prober should have been blocked by the holder, but was not (this is the exact race that caused the production incident)"

wait "${holder_pid}"
echo "PASS: Test 1 (concurrent holder correctly blocks a probing flock -n)"

# --- Test 2: after the holder releases, a probe can acquire cleanly ---
(
  exec 9>"${LOCK_FILE}"
  flock -n 9 || exit 1
  flock -u 9
) || fail "Test 2: prober should acquire cleanly once the holder has released, but did not"
echo "PASS: Test 2 (lock is acquirable again after release)"

echo "All compact/recover lock coordination tests passed."
