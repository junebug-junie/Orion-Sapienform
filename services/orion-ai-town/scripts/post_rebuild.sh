#!/usr/bin/env bash
# Pushes the Convex functions from the patched upstream/ after a good
# post-pull rebuild of orion-ai-town (scripts/rebuild_services_from_git_diff.sh
# runs this only once every compose step passed, so the backend is up and the
# deploy gates let this tree ship). pre_rebuild.sh prepared upstream/.
#
# Test hook: AITOWN_TEST_CONVEX_PUSH runs instead of the push.
set -euo pipefail
UPSTREAM="$(cd "$(dirname "$0")/.." && pwd)/upstream"
cd "${UPSTREAM}"
if [[ -n "${AITOWN_TEST_CONVEX_PUSH:-}" ]]; then
  echo "post_rebuild: AITOWN_TEST_CONVEX_PUSH is set -- NOT pushing functions" >&2
  bash -c "${AITOWN_TEST_CONVEX_PUSH}"
elif command -v npx >/dev/null 2>&1; then
  npx convex dev --once
else
  # circe has no host node; node_modules is already in upstream/.
  docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "${UPSTREAM}:/app" -w /app node:20-alpine npx convex dev --once
fi
echo "post_rebuild: Convex functions pushed from upstream/"
