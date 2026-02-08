#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-${HUB_BASE_URL:-http://127.0.0.1:8080}}"
BASE_URL="${BASE_URL%/}"

html=$(curl -fsS "${BASE_URL}/#topic-studio" || curl -fsS "${BASE_URL}/")
app_js_path=$(echo "$html" | rg -o 'src="(/static/js/app\.js\?v=[^"\s]+)' -r '$1' | head -n 1)
if [[ -z "$app_js_path" ]]; then
  echo "FAIL: app.js script tag not found in hub HTML." >&2
  exit 1
fi

app_js_url="${BASE_URL}${app_js_path}"
bundle=$(curl -fsS "${app_js_url}")

if ! rg -q --fixed-string "TOPIC STUDIO" <<<"$bundle"; then
  echo "FAIL: Missing 'TOPIC STUDIO' in ${app_js_url}" >&2
  exit 1
fi
if ! rg -q --fixed-string "SPLIT PANE" <<<"$bundle"; then
  echo "FAIL: Missing 'SPLIT PANE' sentinel in ${app_js_url}" >&2
  exit 1
fi
if rg -q --fixed-string "params is not defined" <<<"$bundle"; then
  echo "FAIL: Crash string 'params is not defined' found in ${app_js_url}" >&2
  exit 1
fi

echo "Topic Studio UI smoke checks passed."
