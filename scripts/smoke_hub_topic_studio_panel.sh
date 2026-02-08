#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-${HUB_BASE_URL:-http://127.0.0.1:8080}}"
BASE_URL="${BASE_URL%/}"

html=$(curl -fsS "${BASE_URL}/")
if ! rg -q 'section id="topic-studio" data-panel="topic-studio"' <<<"$html"; then
  echo "FAIL: topic-studio panel section not found in HTML." >&2
  exit 1
fi
if ! rg -q 'div id="topicStudioRoot"' <<<"$html"; then
  echo "FAIL: topicStudioRoot not found in HTML." >&2
  exit 1
fi

app_js_path=$(echo "$html" | rg -o 'src="(/static/js/app\.js\?v=[^"\s]+)' -r '$1' | head -n 1)
if [[ -z "$app_js_path" ]]; then
  echo "FAIL: app.js script tag not found in HTML." >&2
  exit 1
fi

curl -fsS "${BASE_URL}${app_js_path}" -o /tmp/app.js
docker run --rm -v /tmp:/tmp node:20-alpine node --check /tmp/app.js

echo "Topic Studio panel smoke checks passed."
