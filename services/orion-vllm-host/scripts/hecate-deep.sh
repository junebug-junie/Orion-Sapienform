#!/usr/bin/env bash
# Explicit deep lane; safe_docker_build enforces clean primary main for deployment.
set -euo pipefail
cd "$(dirname "$0")/../../.."
[[ "$(hostname -s)" == hecate ]] || { echo "Hecate only" >&2; exit 1; }
case "${1:-}" in
    stop|down|logs|ps|config|images|--help) ;;
    *)
        [[ "$(findmnt -n -o SOURCE --target /mnt/models)" == /dev/nvme0n1p1 ]] || exit 1
        python3 services/orion-vllm-host/scripts/check_image_storage.py
        ;;
esac
exec scripts/safe_docker_build.sh orion-vllm-host \
    -f services/orion-vllm-host/docker-compose.deep.yml --profile vllm-worker "$@"
