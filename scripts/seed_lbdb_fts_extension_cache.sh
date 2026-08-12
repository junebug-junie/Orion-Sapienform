#!/bin/bash
# Pre-seeds the LadybugDB FTS DuckDB extension into a host cache directory so
# orion-harness-governor's Docker build (services/orion-harness-governor/Dockerfile)
# COPYs it in offline instead of hitting extension.ladybugdb.com at build time.
#
# Why: that origin has had real sustained outages (Cloudflare 522 for 48+ minutes,
# observed 2026-08-12) that block the image build entirely. The build now reads
# from this host-local cache via a docker-compose `additional_contexts` entry
# (HARNESS_LBDB_EXT_CACHE_DIR) instead of running `install-duckdb-extension.mjs`'s
# network install step. The Dockerfile's COPY destination is parameterized by the
# same LBDB_EXT_VERSION/LBDB_EXT_PLATFORM build args this script reads, so a
# version bump only needs the env var changed in one place (services/
# orion-harness-governor/.env), not a hand-edit of the Dockerfile too.
#
# Idempotent: skips the download if a cache entry for the CURRENT version+platform
# already exists and looks sane (tracked via a sidecar .version marker file --
# the cached filename itself can't encode this, since the Dockerfile COPY source
# name is fixed). Re-run any time LBDB_EXT_VERSION/LBDB_EXT_PLATFORM changes; a
# version bump is detected automatically and triggers a re-fetch.
#
# Not committed to git (binary, host-specific path) -- run this once per host.
set -euo pipefail

# Reads HARNESS_LBDB_EXT_CACHE_DIR from the invoking shell first (explicit
# override always wins), then falls back to services/orion-harness-governor/.env
# (this repo's documented config surface for the same key, and what
# `docker compose build` itself reads via --env-file) before the hardcoded
# default -- otherwise a value set only in that .env is invisible here even
# though it governs where docker-compose's `additional_contexts` build context
# actually points.
_load_cache_dir_from_service_env() {
  local script_dir repo_root env_file line
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  repo_root=$(cd "$script_dir/.." && pwd)
  env_file="$repo_root/services/orion-harness-governor/.env"
  [ -f "$env_file" ] || return 1
  line=$(grep -m1 '^HARNESS_LBDB_EXT_CACHE_DIR=' "$env_file" || true)
  [ -n "$line" ] || return 1
  printf '%s' "${line#HARNESS_LBDB_EXT_CACHE_DIR=}"
}

# Portable file size: GNU `stat -c%s` (Linux) vs BSD/macOS `stat -f%z`.
_file_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

VERSION="${LBDB_EXT_VERSION:-0.18.1}"
PLATFORM="${LBDB_EXT_PLATFORM:-linux_amd64}"
CACHE_DIR="${HARNESS_LBDB_EXT_CACHE_DIR:-$(_load_cache_dir_from_service_env || true)}"
CACHE_DIR="${CACHE_DIR:-/mnt/telemetry/duckdb-extensions/staging}"
OUT="$CACHE_DIR/libfts.lbug_extension"
VERSION_MARKER="$OUT.version"
URL="https://extension.ladybugdb.com/v${VERSION}/${PLATFORM}/fts/libfts.lbug_extension"

mkdir -p "$CACHE_DIR"

if [ -s "$OUT" ] && [ "$(_file_size "$OUT")" -gt 1000 ] \
   && [ "$(cat "$VERSION_MARKER" 2>/dev/null || echo)" = "${VERSION}/${PLATFORM}" ]; then
  echo "already cached: $OUT ($(_file_size "$OUT") bytes, version=${VERSION}/${PLATFORM}) -- delete it to force a re-fetch"
  exit 0
fi

tmp="$OUT.tmp"
code=$(curl -sS -m 30 -o "$tmp" -w "%{http_code}" "$URL" || echo "000")
size=$(_file_size "$tmp")

if [ "$code" = "200" ] && [ "$size" -gt 1000 ]; then
  mv "$tmp" "$OUT"
  echo "${VERSION}/${PLATFORM}" > "$VERSION_MARKER"
  echo "fetched: $OUT ($size bytes, version=${VERSION}/${PLATFORM})"
  exit 0
fi

rm -f "$tmp"
echo "fetch failed: http_code=$code size=$size (extension.ladybugdb.com may be down -- retry later)" >&2
exit 1
