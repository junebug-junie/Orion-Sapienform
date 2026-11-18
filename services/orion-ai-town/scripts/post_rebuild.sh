#!/usr/bin/env bash
# Pushes the Convex functions from the patched upstream/ after a good
# post-pull rebuild of orion-ai-town (scripts/rebuild_services_from_git_diff.sh
# runs this only once every compose step passed, so the backend is up and the
# deploy gates let this tree ship). pre_rebuild.sh prepared upstream/.
set -euo pipefail
bash "$(dirname "$0")/run_convex.sh" dev --once
echo "post_rebuild: Convex functions pushed from upstream/"
