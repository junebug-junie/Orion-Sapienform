#!/usr/bin/env bash
# Runs after a good post-pull rebuild of orion-lane-door (scripts/rebuild_services_from_git_diff.sh): the
# doors are up. Moves any circe engine that still publishes its lane port to loopback, so its door can take
# the port. A no-op once every engine has moved (each deploy after the first), and on any host but circe.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 services/orion-lane-door/scripts/move_engines_behind_doors.py
