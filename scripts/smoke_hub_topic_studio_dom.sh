#!/usr/bin/env bash
set -euo pipefail

HUB_URL=${HUB_URL:-http://localhost:8080}

html=$(curl -fsS "${HUB_URL}/")
if ! printf '%s' "$html" | rg -q "hub-panel-host"; then
  echo "hub-panel-host not found in hub HTML." >&2
  exit 1
fi

if ! rg -q "TOPIC STUDIO ACTIVE" services/orion-hub/static/js/app.js; then
  echo "Topic Studio sentinel string not found in app.js." >&2
  exit 1
fi

echo "OK: hub-panel-host present and Topic Studio sentinel string found."
