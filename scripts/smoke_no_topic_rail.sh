#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pattern='topic_rail|topic rail|Topic Rail|/api/topics/summary|/api/topics/drift'

set +e
matches="$(rg -n -i "$pattern" "$repo_root" 2>/dev/null)"
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "[FAIL] Topic Rail legacy references found:"
  echo "$matches"
  exit 1
fi

if [[ $status -eq 1 ]]; then
  echo "[PASS] No Topic Rail legacy references detected."
  exit 0
fi

echo "[FAIL] rg errored while scanning for Topic Rail references"
exit 1
