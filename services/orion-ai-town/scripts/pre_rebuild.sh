#!/usr/bin/env bash
# Rebuilds upstream/ from the pin before every post-pull rebuild of
# orion-ai-town (scripts/rebuild_services_from_git_diff.sh runs this first;
# exit != 0 skips the stack). The frontend image builds from the gitignored
# upstream/, so without this a pull rebuilt it from whatever upstream/ held:
# the live town ran 2026-08-30 code until 2026-10-23. post_rebuild.sh pushes
# the Convex functions once the stack is up.
#
# 1. Save upstream/'s changes (tracked vs HEAD, plus untracked file contents).
# 2. Reset tracked files to the pin and delete files the patches create, so
#    the chain applies from scratch every time (re-applying on a patched tree
#    stops at orion-conversation-proximity.patch).
# 3. Apply the chain.
#
# Override: AITOWN_ALLOW_EXAMPLE_CARDS=1 (deploy with the redacted example
# cards), AITOWN_BACKUP_ROOT (default ~/.orion).
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
# A pin bump also changes package.json and node_modules: do it by hand.
HEAD_REF="$(git -C "${UPSTREAM}" rev-parse HEAD)"
if [[ "${HEAD_REF}" != "${REF}" ]]; then
  echo "pre_rebuild: upstream HEAD ${HEAD_REF} is not the pin ${REF} -- after" \
    "'git -C upstream checkout --detach ${REF}' and 'npm ci', rerun" >&2
  exit 1
fi

BACKUP_ROOT="${AITOWN_BACKUP_ROOT:-${HOME}/.orion}"
BACKUP="${BACKUP_ROOT}/aitown-upstream-backup-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${BACKUP}/untracked"
git -C "${UPSTREAM}" diff HEAD > "${BACKUP}/tracked.diff"
git -C "${UPSTREAM}" status --short > "${BACKUP}/status.txt"
(cd "${UPSTREAM}" && git ls-files --others --exclude-standard -z | xargs -0 -r cp --parents -t "${BACKUP}/untracked")
# Keep the newest 10.
ls -1d "${BACKUP_ROOT}"/aitown-upstream-backup-* | sort -r | tail -n +11 | while read -r old; do
  rm -r -- "${old}"
done
echo "pre_rebuild: saved upstream changes to ${BACKUP}"

git -C "${UPSTREAM}" checkout -q "${REF}" -- .
for patch in "${ROOT}"/patches/*.patch; do
  awk '/^diff --git /{f=substr($4,3)} /^new file mode/{print f}' "${patch}"
done | while read -r created; do
  rm -f -- "${UPSTREAM:?}/${created}"
done
bash "${ROOT}/scripts/apply_upstream_patches.sh"
echo "pre_rebuild: upstream/ is the pin plus the patch chain"
