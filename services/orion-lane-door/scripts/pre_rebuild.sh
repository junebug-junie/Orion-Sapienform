#!/usr/bin/env bash
# Runs before every post-pull rebuild of orion-lane-door (scripts/rebuild_services_from_git_diff.sh).
# The door's .env holds no secret, so the first deploy on a host makes it from the template: nobody has
# to create it by hand. An existing .env is never touched (sync_local_env_from_example.py adds new keys).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
    cp .env_example .env
    echo "pre_rebuild: services/orion-lane-door/.env created from .env_example"
fi
