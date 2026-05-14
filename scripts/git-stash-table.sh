#!/usr/bin/env bash
# Print git stashes as a table with commit timestamps (UTC from git object).
#
# Usage:
#   ./scripts/git-stash-table.sh           # markdown table (default)
#   ./scripts/git-stash-table.sh --tsv     # tab-separated
#   ./scripts/git-stash-table.sh --plain # column-aligned (no markdown)
#
# Run from any directory inside a git work tree.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: git-stash-table.sh [--tsv|--plain|-h|--help]

  (default)   Markdown table: Stash | When (UTC) | Branch | Message
  --tsv       Tab-separated values
  --plain     column(1)-aligned text
EOF
}

MODE=markdown
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tsv) MODE=tsv ;;
    --plain) MODE=plain ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "error: not inside a git repository" >&2
  exit 1
fi
cd "$REPO_ROOT"

# Args: full subject from "git log -1 --format=%s" on stash ref
# Prints: branch<TAB>message_tail (branch may be empty)
parse_stash_subject() {
  local subj="$1"
  if [[ "$subj" == "On "* ]]; then
    local rest="${subj#On }"
    printf '%s\t%s' "${rest%%: *}" "${rest#*: }"
  elif [[ "$subj" == "WIP on "* ]]; then
    local rest="${subj#WIP on }"
    printf '%s\t%s' "${rest%%: *}" "${rest#*: }"
  else
    printf '\t%s' "$subj"
  fi
}

md_escape() {
  local s="$1"
  s="${s//|/\\|}"
  s="${s//\`/\\\`}"
  printf '%s' "$s"
}

i=0
rows=0
plain_tmp=""
if [[ "$MODE" == "plain" ]]; then
  plain_tmp="$(mktemp)"
  trap 'rm -f "${plain_tmp:-}"' EXIT
fi

if [[ "$MODE" == "markdown" ]]; then
  echo "| Stash | When (UTC) | Branch | Message |"
  echo "|-------|------------|--------|---------|"
elif [[ "$MODE" == "tsv" ]]; then
  printf '%s\t%s\t%s\t%s\n' "stash" "when_utc" "branch" "message"
fi

while git rev-parse "stash@{$i}" &>/dev/null; do
  ref="stash@{$i}"
  when="$(git log -1 --format='%ci' "$ref")"
  subj="$(git log -1 --format='%s' "$ref")"
  IFS=$'\t' read -r branch msg <<<"$(parse_stash_subject "$subj")"

  case "$MODE" in
    markdown)
      echo "| \`$ref\` | $when | \`$(md_escape "$branch")\` | $(md_escape "$msg") |"
      ;;
    tsv)
      # single-line message: tabs in subject become spaces
      msg1="${subj//	/ }"
      printf '%s\t%s\t%s\t%s\n' "$ref" "$when" "$branch" "$msg1"
      ;;
    plain)
      msg1="${subj//	/ }"
      printf '%s\t%s\t%s\t%s\n' "$ref" "$when" "$branch" "$msg1" >>"$plain_tmp"
      ;;
  esac

  rows=$((rows + 1))
  i=$((i + 1))
done

if [[ "$MODE" == "plain" && "$rows" -gt 0 ]]; then
  column -t -s $'\t' "$plain_tmp"
fi

if [[ "$rows" -eq 0 ]]; then
  echo "(no stashes)" >&2
fi
