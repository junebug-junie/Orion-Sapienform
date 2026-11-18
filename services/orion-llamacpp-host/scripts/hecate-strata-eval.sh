#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
[[ "$(hostname -s)" == hecate ]] || { echo 'Hecate only' >&2; exit 1; }
if [[ " $* " == *' up '* ]]; then
  [[ "$(docker inspect -f '{{.State.Running}}' orion-hecate-vllm-host 2>/dev/null || true)" != true ]] || {
    echo 'Reserve agent-deep and stop the vLLM worker before starting Strata.' >&2; exit 1;
  }
  [[ -f /mnt/models/strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf ]] || exit 1
  [[ -f /mnt/models/strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf ]] || exit 1
fi
exec scripts/safe_docker_build.sh orion-llamacpp-host \
  --compose-file docker-compose.hecate-strata-eval.yml "$@"
