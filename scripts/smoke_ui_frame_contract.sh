#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

template_file="$(rg -l 'id="topic-studio"' "$repo_root/services/orion-hub/templates" | head -n 1 || true)"

if [[ -z "$template_file" ]]; then
  echo "[FAIL] Could not find Hub template containing id=\"topic-studio\""
  exit 1
fi

required=(
  'id="hub"'
  'id="topic-studio"'
  '>Hub<'
  '>Topic Studio<'
)

for needle in "${required[@]}"; do
  if ! rg -q "$needle" "$template_file"; then
    echo "[FAIL] Missing required frame contract token: $needle in $template_file"
    exit 1
  fi
done

echo "[PASS] UI frame contract intact in $template_file"
