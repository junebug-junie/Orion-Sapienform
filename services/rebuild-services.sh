#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────
# 🌐  Orion Mesh Auto-Rebuilder (Node-Aware)
# ────────────────────────────────────────────────

# Auto-detect node name & project scope
NODE_NAME=$(hostname -s)
export PROJECT=${PROJECT:-orion@${NODE_NAME}}
export NET=${NET:-app-net}
ROOT_DIR=${ROOT_DIR:-/mnt/services/Orion-Sapienform/services}

echo "🚀 Rebuilding Orion mesh for [$PROJECT] on node [$NODE_NAME]"
echo "📂 Root services directory: $ROOT_DIR"

# ────────────────────────────────────────────────
# ⚙️  Knobs
# ────────────────────────────────────────────────
PRUNE=${PRUNE:-0}
SKIP_BUS=${SKIP_BUS:-0}
MANAGED_REDIS_URL=${MANAGED_REDIS_URL:-}

# Explicit startup priority list (in order)
PRIORITY_SERVICES=(
  orion-bus
  orion-brain
  orion-rag
  orion-hub
  orion-collapse-mirror
)

# ────────────────────────────────────────────────
# 🧹  Teardown (clean stop)
# ────────────────────────────────────────────────
echo "== Teardown =="
for dir in "$ROOT_DIR"/*; do
  [[ -d "$dir" ]] || continue
  if [[ -f "$dir/docker-compose.yml" || -f "$dir/compose.yml" ]]; then
    echo "💤 Stopping $(basename "$dir") ..."
    ( cd "$dir" && docker compose -p "$PROJECT" down --remove-orphans || true )
  fi
done

if [[ "$PRUNE" == "1" ]]; then
  echo "🧽 Pruning old Docker artifacts..."
  docker system prune -af || true
  docker volume prune -f || true
fi

# ────────────────────────────────────────────────
# 🌐  Ensure network exists
# ────────────────────────────────────────────────
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET"
echo "✔ Network ready: $NET"

# ────────────────────────────────────────────────
# 🔑  Managed Redis override
# ────────────────────────────────────────────────
if [[ -n "$MANAGED_REDIS_URL" ]]; then
  export REDIS_URL="$MANAGED_REDIS_URL"
  SKIP_BUS=1
  echo "ℹ Using managed Redis: $MANAGED_REDIS_URL"
fi

# ────────────────────────────────────────────────
# 🧩  Ordered startup
# ────────────────────────────────────────────────
start_service() {
  local svc_dir="$1"
  local svc_name
  svc_name=$(basename "$svc_dir")
  local compose_file

  # find compose file dynamically
  if [[ -f "$svc_dir/compose.yml" ]]; then
    compose_file="$svc_dir/compose.yml"
  elif [[ -f "$svc_dir/docker-compose.yml" ]]; then
    compose_file="$svc_dir/docker-compose.yml"
  else
    echo "⚠️  No compose file for $svc_name, skipping."
    return
  fi

  # Skip orion-bus if SKIP_BUS=1
  if [[ "$SKIP_BUS" == "1" && "$svc_name" == "orion-bus" ]]; then
    echo "ℹ Skipping bus (managed Redis)."
    return
  fi

  echo "🔧 Starting service: $svc_name"
  ( cd "$svc_dir" && docker compose -p "$PROJECT" -f "$compose_file" up -d --build )
}

# 1️⃣ Start priority services in defined order
for svc in "${PRIORITY_SERVICES[@]}"; do
  dir="$ROOT_DIR/$svc"
  [[ -d "$dir" ]] && start_service "$dir"
done

# 2️⃣ Start any remaining services dynamically
echo "🔁 Starting remaining services..."
for dir in "$ROOT_DIR"/*; do
  [[ -d "$dir" ]] || continue
  svc=$(basename "$dir")

  # skip already-started or priority services
  if printf '%s\n' "${PRIORITY_SERVICES[@]}" | grep -qx "$svc"; then
    continue
  fi

  start_service "$dir"
done

echo "✅ All services started for $PROJECT"
