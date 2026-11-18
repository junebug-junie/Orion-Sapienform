#!/usr/bin/env bash
# Runs `npx convex <args>` in upstream/ for pre_rebuild.sh and post_rebuild.sh.
# circe has no host node; node_modules is already in upstream/, so it falls
# back to a node container on the host network.
#
# Test hook: AITOWN_TEST_CONVEX runs instead, with the args as "$@".
set -euo pipefail
UPSTREAM="$(cd "$(dirname "$0")/.." && pwd)/upstream"
cd "${UPSTREAM}"
if [[ -n "${AITOWN_TEST_CONVEX:-}" ]]; then
  echo "run_convex: AITOWN_TEST_CONVEX is set -- NOT running convex $*" >&2
  bash -c "${AITOWN_TEST_CONVEX}" _ "$@"
elif command -v npx >/dev/null 2>&1; then
  npx convex "$@"
else
  docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "${UPSTREAM}:/app" -w /app node:20-alpine npx convex "$@"
fi
