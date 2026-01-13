#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Refresh service env files:
#   services/<svc>/.env_example  ->  services/<svc>/.env
# - No backups
# - Supports excludes (hardcoded + optional exclude file + runtime overrides)
#
# Default exclude file:
#   mesh_utilities/common/exclude_services_env_refresh.txt
#
# Usage:
#   ./mesh_utilities/common/refresh_service_envs.sh
#
# Optional env vars:
#   EXCLUDE_SERVICES="svc1 svc2"          # replaces excludes entirely
#   EXCLUDE_SERVICES_ADD="svc3 svc4"      # adds to excludes
#   EXCLUDE_FILE="path/to/exclude.txt"    # override default exclude file
#
# permissions ephemeral:
# chmod +x mesh-utilities/common/refresh_service_envs.sh
#
# ==============================================================================

SERVICES_DIR="services"
EXAMPLE_NAME=".env_example"
TARGET_NAME=".env"

# --- Default hardcoded excludes (edit these) ---
DEFAULT_EXCLUDES=(
  orion-vision-edge
  orion-security-watcher
)

# --- Locate repo root (script is expected at root/mesh_utilities/common) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# If script moved, try git root fallback
if [[ ! -d "$REPO_ROOT/$SERVICES_DIR" ]]; then
  if command -v git >/dev/null 2>&1; then
    GIT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "${GIT_ROOT:-}" && -d "$GIT_ROOT/$SERVICES_DIR" ]]; then
      REPO_ROOT="$GIT_ROOT"
    fi
  fi
fi

cd "$REPO_ROOT"

if [[ ! -d "$SERVICES_DIR" ]]; then
  echo "ERROR: cannot find $REPO_ROOT/$SERVICES_DIR" >&2
  exit 2
fi

# --- Default exclude file (explicit for env refresh) ---
DEFAULT_EXCLUDE_FILE="$SCRIPT_DIR/exclude_services_env_refresh.txt"
EXCLUDE_FILE="${EXCLUDE_FILE:-$DEFAULT_EXCLUDE_FILE}"

# --- Runtime overrides ---
EXCLUDE_SERVICES="${EXCLUDE_SERVICES:-}"         # replaces excludes entirely
EXCLUDE_SERVICES_ADD="${EXCLUDE_SERVICES_ADD:-}" # append excludes

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
else
  echo "NOTE: exclude file not found (ok): $EXCLUDE_FILE"
fi

# If EXCLUDE_SERVICES is set, it replaces everything
if [[ -n "${EXCLUDE_SERVICES// }" ]]; then
  EFFECTIVE_EXCLUDES=()
  for x in $EXCLUDE_SERVICES; do
    EFFECTIVE_EXCLUDES+=("$x")
  done
fi

# Append more excludes
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

# ------------------------------------------------------------------------------
# Discover candidate services (services/*/.env_example)
# ------------------------------------------------------------------------------
mapfile -t EXAMPLES < <(
  find "$SERVICES_DIR" -mindepth 2 -maxdepth 2 -type f -name "$EXAMPLE_NAME" -print \
  | sort
)

echo "=== Repo root: $REPO_ROOT ==="
echo "=== Exclude file (env refresh): $EXCLUDE_FILE ==="
echo "=== Found $((${#EXAMPLES[@]})) service env examples ($SERVICES_DIR/*/$EXAMPLE_NAME) ==="

if [[ "${#EXAMPLES[@]}" -eq 0 ]]; then
  echo "No $EXAMPLE_NAME files found under $SERVICES_DIR/*"
  exit 0
fi

echo ""
echo "=== Excluded services (effective) ==="
if [[ "${#EFFECTIVE_EXCLUDES[@]}" -gt 0 ]]; then
  printf ' - %s\n' "${EFFECTIVE_EXCLUDES[@]}"
else
  echo " - <none>"
fi

UPDATED=()
SKIPPED_EXCLUDED=()

for ex in "${EXAMPLES[@]}"; do
  svc="$(basename "$(dirname "$ex")")"
  if is_excluded "$svc"; then
    SKIPPED_EXCLUDED+=("$svc")
    continue
  fi

  target="$(dirname "$ex")/$TARGET_NAME"
  cp -f "$ex" "$target"
  UPDATED+=("$svc")
done

echo ""
echo "=== Updated .env files ==="
if [[ "${#UPDATED[@]}" -gt 0 ]]; then
  printf ' - %s\n' "${UPDATED[@]}"
else
  echo " - <none>"
fi

echo ""
echo "=== Skipped (excluded) ==="
if [[ "${#SKIPPED_EXCLUDED[@]}" -gt 0 ]]; then
  printf '%s\n' "${SKIPPED_EXCLUDED[@]}" | sort -u | sed 's/^/ - /'
else
  echo " - <none>"
fi

echo ""
echo "Done. Wrote $TARGET_NAME from $EXAMPLE_NAME for $((${#UPDATED[@]})) services."
