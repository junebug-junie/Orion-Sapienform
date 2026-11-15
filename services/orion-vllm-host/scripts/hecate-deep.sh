#!/usr/bin/env bash
# Explicit deep lane; safe_docker_build enforces clean primary main for deployment.
# The lane door for agent-deep (services/orion-lane-door, lane-door-agent-deep) rides along: `up` starts it
# before the engine, `stop`/`down`/`kill` stop it first (so 8021 is free for another agent-deep engine, e.g.
# the Strata challenger). The engine publishes on 127.0.0.1:18021 (docker-compose.deep.yml); the door takes
# 8021 once the engine answers there. Nothing to set by hand (no key).
set -euo pipefail
cd "$(dirname "$0")/../../.."
[[ "$(hostname -s)" == hecate ]] || { echo "Hecate only" >&2; exit 1; }
if [[ " $* " == *' up '* ]]; then
    [[ "$(docker inspect -f '{{.State.Running}}' orion-hecate-strata-deep-eval 2>/dev/null || true)" != true ]] || {
        echo 'Reserve agent-deep and stop the Strata challenger before starting vLLM.' >&2; exit 1;
    }
fi
case "${1:-}" in
    stop|down|kill)
        scripts/safe_docker_build.sh orion-lane-door --profile hecate stop lane-door-agent-deep || true
        ;;
    logs|ps|config|images|--help) ;;
    *)
        [[ "$(findmnt -n -o SOURCE --target /mnt/models)" == /dev/nvme0n1p1 ]] || exit 1
        python3 services/orion-vllm-host/scripts/check_image_storage.py
        ;;
esac
if [[ " $* " == *' up '* ]]; then
    # Door first: it waits until the engine answers on 127.0.0.1:18021, so the order is safe, and a
    # foreground `up` (no -d) still gets its door. First run on this host: the door's .env is its
    # template (it holds no secret).
    [[ -f services/orion-lane-door/.env ]] || cp services/orion-lane-door/.env_example services/orion-lane-door/.env
    scripts/safe_docker_build.sh orion-lane-door --profile hecate up -d --build lane-door-agent-deep
fi
exec scripts/safe_docker_build.sh orion-vllm-host \
    -f services/orion-vllm-host/docker-compose.deep.yml --profile vllm-worker "$@"
