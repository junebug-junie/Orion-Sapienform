#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-${HUB_BASE_URL:-http://127.0.0.1:8080}}"
BASE_URL="${BASE_URL%/}"

html=$(curl -fsS "${BASE_URL}")
app_js_path=$(echo "$html" | rg -o 'src="(/static/js/app\.js\?v=[^"\s]+)' -r '$1' | head -n 1)
if [[ -z "$app_js_path" ]]; then
  echo "FAIL: Unable to locate app.js script tag in hub HTML." >&2
  exit 1
fi

app_js_url="${BASE_URL}${app_js_path}"
app_js=$(curl -fsS "${app_js_url}")

if command -v node >/dev/null 2>&1; then
  temp_js="$(mktemp)"
  printf "%s" "$app_js" > "$temp_js"
  node --check "$temp_js"
  rm -f "$temp_js"
fi

checks=(
  "TOPIC STUDIO SPLIT PANE v2 ACTIVE"
  "Segment Details"
  "include_full_text"
)

for needle in "${checks[@]}"; do
  if ! rg -q --fixed-string "$needle" <<<"$app_js"; then
    echo "FAIL: Missing '${needle}' in ${app_js_url}" >&2
    exit 1
  fi
done

echo "Topic Studio DOM smoke checks passed."
