#!/usr/bin/env bash
# Apply Orion mesh patches to cloned ai-town upstream before docker build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"
PATCHES=("orion-hub-embed.patch" "orion-character.patch" "orion-engine-recovery.patch")

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  echo "Missing ${UPSTREAM} — clone upstream first (see README.md)" >&2
  exit 1
fi

cd "${UPSTREAM}"
for name in "${PATCHES[@]}"; do
  PATCH="${ROOT}/patches/${name}"
  if [[ ! -f "${PATCH}" ]]; then
    echo "Skipping ${name} (not present)"
    continue
  fi
  # Check "already applied" (reverse) FIRST: additive patches (pure insertions)
  # always pass a forward --check even when already present, so a forward-first
  # order would double-apply them and duplicate the inserted code.
  if git apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "${name} already applied"
  elif git apply --check "${PATCH}" 2>/dev/null; then
    git apply "${PATCH}"
    echo "Applied ${name}"
  else
    echo "${name} does not apply cleanly — resolve upstream drift manually" >&2
    exit 1
  fi
done
