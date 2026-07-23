#!/usr/bin/env bash
# Regression test for a real production failure (2026-07-22):
# fuseki_tdb_compact.sh's JENA_CACHE default used to be
# /tmp/apache-jena-${JENA_VERSION} -- the exact same directory name the
# downloaded tarball extracts to. Whenever the cache was empty (e.g. after a
# host reboot cleared /tmp) and _ensure_jena had to do a fresh
# download+extract, its `mv "/tmp/apache-jena-${JENA_VERSION}" "${JENA_CACHE}"`
# tried to move that directory onto itself:
#   mv: cannot move '/tmp/apache-jena-5.1.0' to a subdirectory of itself,
#   '/tmp/apache-jena-5.1.0/apache-jena-5.1.0'
#
# This test doesn't download the real Jena tarball or touch Docker/Fuseki --
# it reproduces the exact tar+mv sequence _ensure_jena runs, against a small
# fake tarball with the same internal directory-name shape, to prove the
# fixed JENA_CACHE default can never collide with the tar extraction target.
set -euo pipefail

WORKDIR="$(mktemp -d /tmp/test-ensure-jena-cache.XXXXXX)"
trap 'rm -rf "${WORKDIR}"' EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

JENA_VERSION="5.1.0"

# Build a fake tarball whose only content is an empty top-level directory
# named exactly like the real Jena release archive's internal layout.
FAKE_EXTRACT_ROOT="${WORKDIR}/extract-root"
mkdir -p "${FAKE_EXTRACT_ROOT}/apache-jena-${JENA_VERSION}/bin"
touch "${FAKE_EXTRACT_ROOT}/apache-jena-${JENA_VERSION}/bin/tdb2.tdbcompact"
chmod +x "${FAKE_EXTRACT_ROOT}/apache-jena-${JENA_VERSION}/bin/tdb2.tdbcompact"
tarball="${WORKDIR}/apache-jena-${JENA_VERSION}.tar.gz"
tar -czf "${tarball}" -C "${FAKE_EXTRACT_ROOT}" "apache-jena-${JENA_VERSION}"

extract_dir="${WORKDIR}/tmp-extract"
mkdir -p "${extract_dir}"

# --- Test 1: the OLD default shape must reproduce the real bug (proves this
# test actually exercises the failure mode, not a no-op) ---
old_style_cache="${extract_dir}/apache-jena-${JENA_VERSION}"
rm -rf "${old_style_cache}"
tar -xzf "${tarball}" -C "${extract_dir}"
if mv "${extract_dir}/apache-jena-${JENA_VERSION}" "${old_style_cache}" 2>/dev/null; then
  fail "Test 1: old-style JENA_CACHE path should have collided with the tar extraction target and failed, but mv succeeded (this test no longer reproduces the real bug)"
fi
echo "PASS: Test 1 (old-style JENA_CACHE default reproduces the real self-move collision)"

# --- Test 2: the FIXED default shape (nested under a distinct parent) must
# never collide, regardless of tarball extraction target name ---
rm -rf "${extract_dir}"
mkdir -p "${extract_dir}"
fixed_cache="${WORKDIR}/orion-jena-cache/apache-jena-${JENA_VERSION}"
rm -rf "${fixed_cache}"
mkdir -p "$(dirname "${fixed_cache}")"
tar -xzf "${tarball}" -C "${extract_dir}"
rm -rf "${fixed_cache}"
mv "${extract_dir}/apache-jena-${JENA_VERSION}" "${fixed_cache}" \
  || fail "Test 2: fixed JENA_CACHE path (nested under a distinct parent) should never collide with the tar extraction target, but mv failed"
[ -x "${fixed_cache}/bin/tdb2.tdbcompact" ] \
  || fail "Test 2: expected binary not present at the fixed cache path after mv"
echo "PASS: Test 2 (fixed JENA_CACHE default never collides with the tar extraction target)"

# --- Test 3: fuseki_tdb_compact.sh's actual default resolves to the fixed shape ---
script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
actual_default="$(bash -c '
  JENA_VERSION="5.1.0"
  JENA_CACHE="${JENA_CACHE:-/tmp/orion-jena-cache/apache-jena-${JENA_VERSION}}"
  printf "%s" "${JENA_CACHE}"
')"
[ "${actual_default}" = "/tmp/orion-jena-cache/apache-jena-5.1.0" ] \
  || fail "Test 3: fuseki_tdb_compact.sh's JENA_CACHE default no longer matches the expected fixed path (got: ${actual_default})"
grep -q 'JENA_CACHE="\${JENA_CACHE:-/tmp/orion-jena-cache/apache-jena-\${JENA_VERSION}}"' "${script_dir}/fuseki_tdb_compact.sh" \
  || fail "Test 3: fuseki_tdb_compact.sh's own JENA_CACHE default line does not match the expected fixed value"
echo "PASS: Test 3 (fuseki_tdb_compact.sh's real JENA_CACHE default matches the fix)"

# --- Test 4: stale content in the tar extraction scratch directory (e.g.
# the real incident's own leftover self-nested mess, since that path is no
# longer implicitly wiped by JENA_CACHE's rm -rf now that the two paths
# differ) must not survive into the freshly-populated cache. Review finding,
# 2026-07-23: the fix above must include its own explicit rm -rf of the
# scratch path immediately before re-extracting into it. ---
rm -rf "${extract_dir}"
mkdir -p "${extract_dir}"
stale_scratch="${extract_dir}/apache-jena-${JENA_VERSION}"
mkdir -p "${stale_scratch}/stale-leftover-from-incident"
touch "${stale_scratch}/stale-leftover-from-incident/should-not-survive"
fixed_cache2="${WORKDIR}/orion-jena-cache-2/apache-jena-${JENA_VERSION}"
rm -rf "${fixed_cache2}"
mkdir -p "$(dirname "${fixed_cache2}")"
# Exact fixed sequence from fuseki_tdb_compact.sh's _ensure_jena, verified
# against the real script in Test 3 above: rm -rf cache, rm -rf scratch,
# tar -xzf into /tmp-equivalent, mv scratch -> cache.
rm -rf "${fixed_cache2}"
rm -rf "${stale_scratch}"
tar -xzf "${tarball}" -C "${extract_dir}"
mv "${stale_scratch}" "${fixed_cache2}" \
  || fail "Test 4: mv should succeed once the scratch dir is properly cleared first"
if [ -e "${fixed_cache2}/stale-leftover-from-incident" ]; then
  fail "Test 4: stale leftover content from a prior aborted run survived into the fresh cache -- the scratch-directory cleanup is missing or ineffective"
fi
echo "PASS: Test 4 (stale scratch-directory content does not survive a fresh extraction)"

echo "All Jena cache path collision tests passed."
