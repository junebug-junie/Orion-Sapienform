#!/usr/bin/env bash
# Apply Orion mesh patches to cloned ai-town upstream before docker build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"
PATCHES=("orion-hub-embed.patch" "orion-character.patch")

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
  if git apply --check "${PATCH}" 2>/dev/null; then
    git apply "${PATCH}"
    echo "Applied ${name}"
  elif git apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "${name} already applied"
  else
    echo "${name} does not apply cleanly — resolve upstream drift manually" >&2
    exit 1
  fi
done
