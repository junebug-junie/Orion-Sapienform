#!/usr/bin/env bash
# Apply Orion mesh patches to cloned ai-town upstream before docker build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"
PATCH="${ROOT}/patches/orion-hub-embed.patch"

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  echo "Missing ${UPSTREAM} — clone upstream first (see README.md)" >&2
  exit 1
fi
if [[ ! -f "${PATCH}" ]]; then
  echo "Missing patch ${PATCH}" >&2
  exit 1
fi

cd "${UPSTREAM}"
if git apply --check "${PATCH}" 2>/dev/null; then
  git apply "${PATCH}"
  echo "Applied ${PATCH}"
elif git apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "Patch already applied"
else
  echo "Patch does not apply cleanly — resolve upstream drift manually" >&2
  exit 1
fi
