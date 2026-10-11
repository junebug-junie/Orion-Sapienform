#!/usr/bin/env bash
# Explicit build/run path; generic Athena automation never selects this profile.
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [ "$(hostname -s)" != hecate ]; then
    echo "Hecate worker commands must run on Hecate" >&2
    exit 1
fi
if [ "$(findmnt -n -o SOURCE --target /mnt/models)" != /dev/nvme0n1p1 ]; then
    echo "Hecate model NVMe is not mounted at /mnt/models" >&2
    exit 1
fi
exec scripts/safe_docker_build.sh orion-vllm-host \
    -f services/orion-vllm-host/docker-compose.hecate.yml --profile vllm-worker "$@"
