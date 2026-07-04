#!/usr/bin/env bash
# Shared helpers: explicit cortex-exec lane bring-up + post-up verification.
# Sourced by mesh-utilities/common/up_all_services*.sh

CORTEX_EXEC_SERVICE_DIR="orion-cortex-exec"

CORTEX_EXEC_COMPOSE_SERVICES=(
  cortex-exec
  cortex-exec-chat
  cortex-exec-spark
  cortex-exec-background
)

_cortex_exec_project_name() {
  local repo_root="$1"
  local project=""
  if [[ -f "$repo_root/services/$CORTEX_EXEC_SERVICE_DIR/.env" ]]; then
    project="$(grep -E '^PROJECT=' "$repo_root/services/$CORTEX_EXEC_SERVICE_DIR/.env" | tail -1 | cut -d= -f2- | tr -d "\"'" | xargs)"
  fi
  echo "${project:-orion-athena}"
}

# Build + up all compose services in the cortex-exec stack (not just the default service).
up_cortex_exec_fleet() {
  local compose_cmd="$1"
  local repo_root="$2"

  echo ""
  echo "=== [$CORTEX_EXEC_SERVICE_DIR] build + up (all lane containers, explicit) ==="
  set +e
  # shellcheck disable=SC2086
  $compose_cmd up -d --build \
    "${CORTEX_EXEC_COMPOSE_SERVICES[@]}"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi
  verify_cortex_exec_fleet "$repo_root"
}

# Fail if any lane container is missing weights or has the wrong pre-turn handler flag.
verify_cortex_exec_fleet() {
  local repo_root="$1"
  local project
  project="$(_cortex_exec_project_name "$repo_root")"

  local -a cnames=(
    "${project}-cortex-exec"
    "${project}-cortex-exec-chat"
    "${project}-cortex-exec-spark"
    "${project}-cortex-exec-background"
  )
  local -a expect_handler=(true false false false)
  local failures=0

  echo ""
  echo "=== [$CORTEX_EXEC_SERVICE_DIR] fleet verification (project=$project) ==="

  local i=0
  for cname in "${cnames[@]}"; do
    local expect="${expect_handler[$i]}"
    i=$((i + 1))

    if ! docker inspect "$cname" >/dev/null 2>&1; then
      echo "❌ $cname: container missing"
      failures=$((failures + 1))
      continue
    fi

    local status
    status="$(docker inspect -f '{{.State.Status}}' "$cname" 2>/dev/null || echo unknown)"
    if [[ "$status" != "running" ]]; then
      echo "❌ $cname: status=$status (expected running)"
      failures=$((failures + 1))
      continue
    fi

    if ! docker exec "$cname" test -f /app/config/substrate/repair_pressure_weights.v2.yaml 2>/dev/null; then
      echo "❌ $cname: missing /app/config/substrate/repair_pressure_weights.v2.yaml"
      failures=$((failures + 1))
      continue
    fi

    local handler
    handler="$(docker exec "$cname" printenv ENABLE_PRE_TURN_APPRAISAL_HANDLER 2>/dev/null || true)"
    handler="${handler,,}"
    if [[ "$expect" == "true" ]]; then
      if [[ "$handler" == "false" || "$handler" == "0" ]]; then
        echo "❌ $cname: ENABLE_PRE_TURN_APPRAISAL_HANDLER=$handler (main must handle pre-turn RPC)"
        failures=$((failures + 1))
        continue
      fi
    else
      if [[ -n "$handler" && "$handler" != "false" && "$handler" != "0" ]]; then
        echo "❌ $cname: ENABLE_PRE_TURN_APPRAISAL_HANDLER=$handler (lane must not handle pre-turn RPC)"
        failures=$((failures + 1))
        continue
      fi
    fi

    echo "✅ $cname: running, weights OK, pre_turn_handler=$expect"
  done

  if [[ "$failures" -gt 0 ]]; then
    echo "ERROR: cortex-exec fleet verification failed ($failures issue(s))" >&2
    return 1
  fi
  return 0
}
