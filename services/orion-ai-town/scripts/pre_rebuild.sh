#!/usr/bin/env bash
# Deploys the town's code before every post-pull rebuild of orion-ai-town
# (scripts/rebuild_services_from_git_diff.sh runs this first; exit != 0 skips
# the stack). The frontend image builds from the gitignored upstream/ and the
# Convex functions live outside compose, so a plain rebuild shipped neither:
# the live town ran 2026-08-30 code until 2026-10-23.
#
# 1. Save upstream/'s current diff (hand edits are recoverable).
# 2. Reset upstream/ to the pin and delete files the patches create, so the
#    chain applies from scratch every time (re-applying on a patched tree
#    stops at orion-conversation-proximity.patch).
# 3. Apply the chain. 4. Push the Convex functions.
#
# Overrides: AITOWN_CONVEX_PUSH (command run in upstream/ instead of the push),
# AITOWN_ALLOW_EXAMPLE_CARDS=1 (deploy with the redacted example cards),
# AITOWN_BACKUP_ROOT (default ~/.orion).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  echo "pre_rebuild: missing ${UPSTREAM} -- clone upstream first (README.md)" >&2
  exit 1
fi
# Without the local cards the Juniper join text renders from the redacted
# example and replaces the real one in the live world (circe, 2026-10-23).
if [[ ! -f "${ROOT}/cards/town_cards.local.yaml" && "${AITOWN_ALLOW_EXAMPLE_CARDS:-0}" != "1" ]]; then
  echo "pre_rebuild: cards/town_cards.local.yaml missing -- copy it from the operator host" \
    "(owner-only, gitignored) or set AITOWN_ALLOW_EXAMPLE_CARDS=1" >&2
  exit 1
fi
REF="$(sed -n 's/^AITOWN_UPSTREAM_REF=\([0-9a-f]\{40\}\)$/\1/p' "${ROOT}/.env_example")"
if [[ -z "${REF}" ]]; then
  echo "pre_rebuild: no 40-hex AITOWN_UPSTREAM_REF in .env_example" >&2
  exit 1
fi

BACKUP_ROOT="${AITOWN_BACKUP_ROOT:-${HOME}/.orion}"
BACKUP="${BACKUP_ROOT}/aitown-upstream-backup-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${BACKUP}"
git -C "${UPSTREAM}" diff > "${BACKUP}/tracked.diff"
git -C "${UPSTREAM}" status --short > "${BACKUP}/status.txt"
# Keep the newest 10.
ls -1d "${BACKUP_ROOT}"/aitown-upstream-backup-* 2>/dev/null | sort | head -n -10 | while read -r old; do
  rm -r -- "${old}"
done
echo "pre_rebuild: saved upstream diff to ${BACKUP}"

git -C "${UPSTREAM}" checkout -q "${REF}" -- .
for patch in "${ROOT}"/patches/*.patch; do
  awk '/^diff --git /{f=substr($4,3)} /^new file mode/{print f}' "${patch}"
done | while read -r created; do
  rm -f -- "${UPSTREAM:?}/${created}"
done
bash "${ROOT}/scripts/apply_upstream_patches.sh"

cd "${UPSTREAM}"
if [[ -n "${AITOWN_CONVEX_PUSH:-}" ]]; then
  bash -c "${AITOWN_CONVEX_PUSH}"
elif command -v npx >/dev/null 2>&1; then
  npx convex dev --once
else
  # circe has no host node; node_modules is already in upstream/.
  docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "${UPSTREAM}:/app" -w /app node:20-alpine npx convex dev --once
fi
echo "pre_rebuild: town code deployed at ${REF}"
