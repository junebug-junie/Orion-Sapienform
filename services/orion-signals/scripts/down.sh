#!/usr/bin/env bash
# Orion Signals mesh launcher — reverse-order tier teardown.
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

TIER_ARG="${1:-${SIGNALS_TIER:-core}}"
SIGNALS_USE_BUNDLED_REDIS="${SIGNALS_USE_BUNDLED_REDIS:-false}"

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

for entry in reversed(entries):
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

STOPPED=()

compose_down() {
  local compose_dir="$1"
  shift
  local -a services=("$@")
  local compose_file="${REPO_ROOT}/services/${compose_dir}/docker-compose.yml"
  local service_env="${REPO_ROOT}/services/${compose_dir}/.env"

  if [[ ! -f "${compose_file}" ]]; then
    echo "WARN: missing compose file: ${compose_file} — skipping" >&2
    return 0
  fi

  local -a cmd=(
    docker compose
    --env-file "${service_env}"
    --env-file "${ENV_FILE}"
    -f "${compose_file}"
    stop
  )
  cmd+=("${services[@]}")

  echo "→ docker compose -f services/${compose_dir}/docker-compose.yml stop ${services[*]}"
  "${cmd[@]}" || true
}

for line in "${SERVICE_LINES[@]}"; do
  IFS=$'\t' read -r svc_id compose_dir compose_service <<<"${line}"

  if [[ "${svc_id}" == "orion-signal-gateway" ]]; then
    compose_down "${compose_dir}" orion-signal-gateway
    compose_down "${compose_dir}" otel-grafana otel-collector otel-tempo
    if [[ "${SIGNALS_USE_BUNDLED_REDIS}" == "true" ]]; then
      compose_down "${compose_dir}" orion-redis
    fi
    STOPPED+=("orion-signal-gateway stack")
  else
    compose_down "${compose_dir}" "${compose_service}"
    STOPPED+=("${svc_id}")
  fi
done

echo ""
echo "=== Orion Signals mesh stopped (tier=${TIER_ARG}) ==="
for item in "${STOPPED[@]}"; do
  echo "  ✓ ${item}"
done
