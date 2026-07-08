#!/usr/bin/env bash
# Orion Signals mesh launcher — cumulative tier bring-up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIGNALS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SIGNALS_DIR}/../.." && pwd)"
ROSTER_FILE="${SIGNALS_DIR}/roster.v1.yaml"
ENV_FILE="${SIGNALS_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${ORION_BUS_URL:-}" ]]; then
  echo "ERROR: ORION_BUS_URL is not set." >&2
  echo "Set it in ${ENV_FILE} (see .env_example)." >&2
  echo "MUST be redis://<tailscale-node-ip>:6379/0 for mesh bus." >&2
  exit 1
fi

TIER_ARG="${1:-${SIGNALS_TIER:-core}}"
SIGNALS_USE_BUNDLED_REDIS="${SIGNALS_USE_BUNDLED_REDIS:-false}"
PROJECT="${PROJECT:-orion-athena}"

export ORION_BUS_URL PROJECT NODE_NAME SIGNAL_GATEWAY_HTTP_PORT

mapfile -t SERVICE_LINES < <(
  python3 - "${ROSTER_FILE}" "${TIER_ARG}" <<'PY'
import sys

try:
    import yaml
except ImportError:
    sys.stderr.write("ERROR: PyYAML required. pip install pyyaml or use repo venv.\n")
    sys.exit(1)

roster_path, tier = sys.argv[1:3]
with open(roster_path, encoding="utf-8") as fh:
    doc = yaml.safe_load(fh)

if doc.get("schema_version") != "orion_signals_roster.v1":
    sys.stderr.write(f"ERROR: unsupported roster schema: {doc.get('schema_version')}\n")
    sys.exit(1)

tier_order = ["core", "tier1", "tier2", "routing"]
tier_idx = {
    "core": 0,
    "tier1": 1,
    "tier2": 2,
    "routing": 3,
    "full": 4,
}
if tier not in tier_idx:
    sys.stderr.write(f"ERROR: unknown tier '{tier}'. Use: core|tier1|tier2|routing|full\n")
    sys.exit(1)

target = tier_idx[tier]
seen: set[str] = set()
entries: list[dict] = []

if tier == "routing":
    for name in ("routing",):
        for entry in doc.get(name, []) or []:
            eid = entry["id"]
            if eid not in seen:
                seen.add(eid)
                entries.append(entry)
else:
    for name in tier_order:
        if tier_order.index(name) > target and tier != "full":
            break
        for entry in doc.get(name, []) or []:
            eid = entry["id"]
            if eid not in seen:
                seen.add(eid)
                entries.append(entry)

for entry in entries:
    print(
        "\t".join(
            [
                entry["id"],
                entry["compose_dir"],
                entry["compose_service"],
            ]
        )
    )
PY
)

if [[ ${#SERVICE_LINES[@]} -eq 0 ]]; then
  echo "ERROR: no services resolved for tier '${TIER_ARG}'" >&2
  exit 1
fi

STARTED=()

compose_up() {
  local compose_dir="$1"
  shift
  local -a services=("$@")
  local compose_file="${REPO_ROOT}/services/${compose_dir}/docker-compose.yml"
  local service_env="${REPO_ROOT}/services/${compose_dir}/.env"

  if [[ ! -f "${compose_file}" ]]; then
    echo "ERROR: missing compose file: ${compose_file}" >&2
    exit 1
  fi

  local -a cmd=(
    docker compose
    --env-file "${service_env}"
    --env-file "${ENV_FILE}"
    -f "${compose_file}"
    up -d
  )
  cmd+=("${services[@]}")

  echo "→ docker compose -f services/${compose_dir}/docker-compose.yml up -d ${services[*]}"
  "${cmd[@]}"
}

for line in "${SERVICE_LINES[@]}"; do
  IFS=$'\t' read -r svc_id compose_dir compose_service <<<"${line}"

  if [[ "${svc_id}" == "orion-signal-gateway" ]]; then
    if [[ "${SIGNALS_USE_BUNDLED_REDIS}" == "true" ]]; then
      compose_up "${compose_dir}" \
        orion-redis \
        otel-tempo \
        otel-collector \
        otel-grafana \
        orion-signal-gateway
      STARTED+=("orion-signal-gateway (+ bundled redis, otel stack)")
    else
      compose_up "${compose_dir}" otel-tempo
      compose_up "${compose_dir}" otel-collector otel-grafana
      docker compose \
        --env-file "${REPO_ROOT}/services/${compose_dir}/.env" \
        --env-file "${ENV_FILE}" \
        -f "${REPO_ROOT}/services/${compose_dir}/docker-compose.yml" \
        up -d --no-deps orion-signal-gateway
      echo "→ docker compose -f services/${compose_dir}/docker-compose.yml up -d --no-deps orion-signal-gateway"
      STARTED+=("orion-signal-gateway (+ otel stack, no bundled redis)")
    fi
  else
    compose_up "${compose_dir}" "${compose_service}"
    STARTED+=("${svc_id} (${compose_service})")
  fi
done

echo ""
echo "=== Orion Signals mesh started (tier=${TIER_ARG}) ==="
for item in "${STARTED[@]}"; do
  echo "  ✓ ${item}"
done
echo "ORION_BUS_URL=${ORION_BUS_URL}"
