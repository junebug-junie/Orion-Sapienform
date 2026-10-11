#!/usr/bin/env bash
# Explicit build/run path; generic Athena automation never selects this profile.
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [ "$(hostname -s)" != hecate ]; then
    echo "Hecate worker commands must run on Hecate" >&2
    exit 1
fi
case "${1:-}" in
    stop|down|kill|logs|ps|config|images|version|--help)
        ;; # Storage failures must not obstruct inspection or shutdown.
    *)
        if [ "$(findmnt -n -o SOURCE --target /mnt/models)" != /dev/nvme0n1p1 ]; then
            echo "Hecate model NVMe is not mounted at /mnt/models" >&2
            exit 1
        fi
        python3 services/orion-vllm-host/scripts/check_image_storage.py
        ;;
esac
exec scripts/safe_docker_build.sh orion-vllm-host \
    -f services/orion-vllm-host/docker-compose.hecate.yml --profile vllm-worker "$@"
