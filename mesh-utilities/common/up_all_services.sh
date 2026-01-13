#!/usr/bin/env bash
set -u
set -o pipefail

# ==============================================================================
# Orion Mesh - Bring up all service docker-compose stacks (bus first)
#
# IMPORTANT:
# - This script intentionally does NOT set a docker compose project name (-p).
#   Your services already manage stable naming via .env + container_name/image tags.
#   Forcing -p causes orphan warnings + container_name conflicts (double-prefixes).
#
# Usage:
#   ./mesh-utilities/common/up_all_services.sh
#
# Optional env vars:
#   WAIT_SEC=30
#   PROJECT_NAME=orion                  # informational only (NOT used for -p)
#   EXCLUDE_SERVICES="svc1 svc2"        # replaces defaults/file excludes
#   EXCLUDE_SERVICES_ADD="svc3 svc4"    # adds to excludes
#   EXCLUDE_FILE="path/to/exclude.txt"  # defaults to mesh-utilities/common/exclude_services.txt
#
# permissions ephemeral:
#   chmod +x mesh-utilities/common/up_all_services.sh
# permissions git persisted:
#   git add --chmod=+x mesh-utilities/common/up_all_services.sh
# ==============================================================================

WAIT_SEC="${WAIT_SEC:-30}"
PROJECT_NAME="${PROJECT_NAME:-${COMPOSE_PROJECT_NAME:-${PROJECT:-orion}}}"  # informational only

BUS_SERVICE_DIR="orion-bus"

SERVICES_DIR="services"
COMPOSE_BASENAME="docker-compose.yml"
ROOT_ENV=".env"

# --- Default hardcoded excludes (edit these) ---
DEFAULT_EXCLUDES=(
  orion-vision-edge
  orion-security-watcher
)

# --- Optional exclude file (one service name per line; supports # comments) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXCLUDE_FILE="$SCRIPT_DIR/exclude_services.txt"
EXCLUDE_FILE="${EXCLUDE_FILE:-$DEFAULT_EXCLUDE_FILE}"

# --- Optional runtime overrides ---
EXCLUDE_SERVICES="${EXCLUDE_SERVICES:-}"         # replace excludes entirely
EXCLUDE_SERVICES_ADD="${EXCLUDE_SERVICES_ADD:-}" # append excludes

# --- repo root detection (script is expected at root/mesh-utilities/common) ---
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ ! -d "$REPO_ROOT/$SERVICES_DIR" ]]; then
  if command -v git >/dev/null 2>&1; then
    GIT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "${GIT_ROOT:-}" && -d "$GIT_ROOT/$SERVICES_DIR" ]]; then
      REPO_ROOT="$GIT_ROOT"
    fi
  fi
fi

cd "$REPO_ROOT" || exit 2

if [[ ! -f "$ROOT_ENV" ]]; then
  echo "ERROR: missing $REPO_ROOT/$ROOT_ENV (run from repo root or ensure it exists)." >&2
  exit 2
fi

# ------------------------------------------------------------------------------
# Build effective excludes
# ------------------------------------------------------------------------------
EFFECTIVE_EXCLUDES=()
EFFECTIVE_EXCLUDES+=("${DEFAULT_EXCLUDES[@]}")

if [[ -f "$EXCLUDE_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"             # strip comments
    line="$(echo "$line" | xargs)" # trim
    [[ -z "$line" ]] && continue
    EFFECTIVE_EXCLUDES+=("$line")
  done < "$EXCLUDE_FILE"
fi

# If EXCLUDE_SERVICES is set, it replaces everything
if [[ -n "${EXCLUDE_SERVICES// }" ]]; then
  EFFECTIVE_EXCLUDES=()
  for x in $EXCLUDE_SERVICES; do
    EFFECTIVE_EXCLUDES+=("$x")
  done
fi

# Append more
if [[ -n "${EXCLUDE_SERVICES_ADD// }" ]]; then
  for x in $EXCLUDE_SERVICES_ADD; do
    EFFECTIVE_EXCLUDES+=("$x")
  done
fi

is_excluded() {
  local svc="$1"
  for x in "${EFFECTIVE_EXCLUDES[@]}"; do
    [[ "$svc" == "$x" ]] && return 0
  done
  return 1
}

compose_cmd() {
  local svc="$1"
  local compose_file="$SERVICES_DIR/$svc/$COMPOSE_BASENAME"
  local svc_env="$SERVICES_DIR/$svc/.env"

  if [[ ! -f "$compose_file" ]]; then
    echo "ERROR: missing $compose_file" >&2
    return 2
  fi
  if [[ ! -f "$svc_env" ]]; then
    echo "ERROR: missing $svc_env" >&2
    return 2
  fi

  # IMPORTANT: no "-p ..." here. Must match your manual invocation pattern.
  echo "docker compose --env-file $ROOT_ENV --env-file $svc_env -f $compose_file"
}

up_one() {
  local svc="$1"
  local cmd
  cmd="$(compose_cmd "$svc")" || return 2

  echo ""
  echo "=== [$svc] build + up ==="
  # Don't let one failure abort the whole run; collect failures.
  set +e
  # shellcheck disable=SC2086
  $cmd up -d --build
  local rc=$?
  set -e
  return $rc
}

ps_ids() {
  local svc="$1"
  local cmd
  cmd="$(compose_cmd "$svc")" || return 0
  set +e
  # shellcheck disable=SC2086
  $cmd ps -q 2>/dev/null
  set -e
}

print_failed_logs() {
  local cid="$1"
  echo "---- logs (tail 80) for container $cid ----"
  docker logs --tail 80 "$cid" 2>&1 || true
}

# ------------------------------------------------------------------------------
# Discover candidates
# ------------------------------------------------------------------------------
set -e

mapfile -t CANDIDATES < <(
  find "$SERVICES_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
  | sort \
  | while read -r svc; do
      [[ -f "$SERVICES_DIR/$svc/$COMPOSE_BASENAME" ]] && echo "$svc"
    done
)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "ERROR: No services found under $SERVICES_DIR/*/$COMPOSE_BASENAME" >&2
  exit 2
fi

echo "=== Repo root: $REPO_ROOT ==="
echo "=== Project name (informational): $PROJECT_NAME ==="
echo "=== Discovered services (have $COMPOSE_BASENAME) ==="
printf ' - %s\n' "${CANDIDATES[@]}"

echo ""
echo "=== Excluded services (effective) ==="
if [[ "${#EFFECTIVE_EXCLUDES[@]}" -gt 0 ]]; then
  printf ' - %s\n' "${EFFECTIVE_EXCLUDES[@]}"
else
  echo " - <none>"
fi

# Filter services we will run (must have .env and not excluded)
RUN_LIST=()
SKIPPED_NOENV=()
SKIPPED_EXCLUDED=()

for svc in "${CANDIDATES[@]}"; do
  if is_excluded "$svc"; then
    SKIPPED_EXCLUDED+=("$svc")
    continue
  fi
  if [[ ! -f "$SERVICES_DIR/$svc/.env" ]]; then
    SKIPPED_NOENV+=("$svc")
    continue
  fi
  RUN_LIST+=("$svc")
done

echo ""
echo "=== Will run (after filtering) ==="
printf ' - %s\n' "${RUN_LIST[@]}"

if [[ "${#SKIPPED_EXCLUDED[@]}" -gt 0 ]]; then
  echo ""
  echo "=== Skipped (excluded) ==="
  printf ' - %s\n' "${SKIPPED_EXCLUDED[@]}"
fi

if [[ "${#SKIPPED_NOENV[@]}" -gt 0 ]]; then
  echo ""
  echo "=== Skipped (missing services/<svc>/.env) ==="
  printf ' - %s\n' "${SKIPPED_NOENV[@]}"
fi

# ------------------------------------------------------------------------------
# Bring up: bus first, then rest
# ------------------------------------------------------------------------------
FAILED_UP=()

if [[ " ${RUN_LIST[*]} " =~ " ${BUS_SERVICE_DIR} " ]]; then
  if ! up_one "$BUS_SERVICE_DIR"; then
    FAILED_UP+=("$BUS_SERVICE_DIR")
  fi
else
  echo ""
  echo "WARNING: $BUS_SERVICE_DIR not in run list (missing compose/.env or excluded)."
fi

for svc in "${RUN_LIST[@]}"; do
  [[ "$svc" == "$BUS_SERVICE_DIR" ]] && continue
  if ! up_one "$svc"; then
    FAILED_UP+=("$svc")
  fi
done

echo ""
echo "=== Waiting $WAIT_SEC seconds for containers to settle... ==="
sleep "$WAIT_SEC"

# ------------------------------------------------------------------------------
# Status report
# ------------------------------------------------------------------------------
echo ""
echo "=== STATUS REPORT (informational project: $PROJECT_NAME) ==="
printf "%-35s  %-12s  %s\n" "service" "running/total" "details"
printf "%-35s  %-12s  %s\n" "-------" "------------" "-------"

FAILED_SERVICES=()

for svc in "${RUN_LIST[@]}"; do
  ids="$(ps_ids "$svc" || true)"

  if [[ -z "${ids//[$'\n\r\t ']}" ]]; then
    printf "%-35s  %-12s  %s\n" "$svc" "0/0" "NO CONTAINERS (compose may have failed)"
    FAILED_SERVICES+=("$svc")
    continue
  fi

  total=0
  running=0
  details=()

  while IFS= read -r cid; do
    [[ -z "$cid" ]] && continue
    total=$((total+1))
    status="$(docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || echo "unknown")"
    exitcode="$(docker inspect -f '{{.State.ExitCode}}' "$cid" 2>/dev/null || echo "?")"
    if [[ "$status" == "running" ]]; then
      running=$((running+1))
    else
      details+=("$cid:$status(exit=$exitcode)")
    fi
  done <<< "$ids"

  if [[ "$running" -eq "$total" ]]; then
    printf "%-35s  %-12s  %s\n" "$svc" "$running/$total" "OK"
  else
    printf "%-35s  %-12s  %s\n" "$svc" "$running/$total" "FAIL"
    FAILED_SERVICES+=("$svc")

    # Tail logs for non-running containers
    while IFS= read -r cid; do
      [[ -z "$cid" ]] && continue
      status="$(docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || echo "unknown")"
      if [[ "$status" != "running" ]]; then
        print_failed_logs "$cid"
      fi
    done <<< "$ids"
  fi
done

echo ""
if [[ "${#FAILED_UP[@]}" -gt 0 ]]; then
  echo "⚠️  Compose up failures during bring-up:"
  printf ' - %s\n' "${FAILED_UP[@]}"
  echo ""
fi

if [[ "${#FAILED_SERVICES[@]}" -eq 0 ]]; then
  echo "✅ All services are running."
else
  echo "❌ Services with issues:"
  printf ' - %s\n' "${FAILED_SERVICES[@]}"
  exit 1
fi
