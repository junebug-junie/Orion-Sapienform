#!/usr/bin/env bash
# Pre-flight for mesh skills: host must have tailscale + tailscaled socket. docker-compose.yml
# already includes bind-mounts; this script only validates paths then `docker compose up`.
#
#   cd services/orion-cortex-exec && ./scripts/up-with-tailscale.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORTEX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CORTEX_DIR"

BIN="${ORION_HOST_TAILSCALE_BIN:-}"
if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
  for cand in /usr/bin/tailscale /usr/sbin/tailscale /snap/bin/tailscale; do
    if [[ -f "${cand}" && -x "${cand}" ]]; then
      BIN="${cand}"
      break
    fi
  done
fi
if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
  echo "up-with-tailscale: no executable tailscale on the host (tried /usr/bin, /usr/sbin, /snap/bin)." >&2
  echo "Install Tailscale on the host, or set ORION_HOST_TAILSCALE_BIN to the real path." >&2
  exit 1
fi

RUN="${ORION_HOST_TAILSCALE_RUN:-/var/run/tailscale}"
if [[ ! -e "${RUN}" ]]; then
  echo "up-with-tailscale: host path missing: ${RUN} (tailscaled socket dir)." >&2
  echo "Ensure tailscaled is running on the host." >&2
  exit 1
fi

export ORION_HOST_TAILSCALE_BIN="${BIN}"
export ORION_HOST_TAILSCALE_RUN="${RUN}"
export ORION_CONTAINER_TAILSCALE_BIN="${ORION_CONTAINER_TAILSCALE_BIN:-/usr/bin/tailscale}"
export ORION_ACTIONS_TAILSCALE_PATH="${ORION_CONTAINER_TAILSCALE_BIN}"

ENV_FILES=()
[[ -f .env ]] && ENV_FILES+=(--env-file .env)
[[ -f ../../.env ]] && ENV_FILES+=(--env-file ../../.env)

echo "up-with-tailscale: using host CLI ${BIN} -> container ${ORION_CONTAINER_TAILSCALE_BIN}, socket ${RUN}"
exec docker compose "${ENV_FILES[@]}" -f docker-compose.yml up -d cortex-exec
