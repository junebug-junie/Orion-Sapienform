#!/usr/bin/env bash
# Apply Orion mesh patches to cloned ai-town upstream before docker build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"
PATCHES=(
  "orion-hub-embed.patch"
  "orion-character.patch"
  "orion-human-juniper.patch"
  "orion-engine-recovery.patch"
  "orion-conversation-proximity.patch"
  "orion-town-chat-turns.patch"
  "orion-npc-cooldown-tuning.patch"
  "orion-anti-repetition-prompt.patch"
  "orion-concrete-grounding-prompt.patch"
  "orion-town-continuity-ingest.patch"
  "orion-input-counter-contention.patch"
  "orion-no-human-idle-kick.patch"
  "orion-npc-answer-first.patch"
  "orion-resync-agent-descriptions.patch"
  "orion-npc-no-self-repeat.patch"
  "orion-mechanical-leave.patch"
  "orion-live-turn-roles.patch"
)

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  echo "Missing ${UPSTREAM} — clone upstream first (see README.md)" >&2
  exit 1
fi

cd "${UPSTREAM}"
for name in "${PATCHES[@]}"; do
  PATCH="${ROOT}/patches/${name}"
  if [[ ! -f "${PATCH}" ]]; then
    echo "Skipping ${name} (not present)"
    continue
  fi
  # The committed human-join patch holds a token in place of Juniper's join
  # description (personal details stay out of the public repo). Render it from
  # the cards -- town_cards.local.yaml first, the redacted example second.
  if [[ "${name}" == "orion-human-juniper.patch" ]]; then
    if [[ ! -f "${ROOT}/cards/town_cards.local.yaml" ]]; then
      echo "note: cards/town_cards.local.yaml missing -- rendering the redacted example card;" \
        "an upstream already patched with the real card will not match" >&2
    fi
    RENDERED="$(mktemp)"
    trap 'rm -f "${RENDERED:-}"' EXIT
    if ! python3 "${ROOT}/scripts/generate_descriptions.py" --render-human-patch > "${RENDERED}"; then
      echo "${name}: rendering from the cards failed (needs python3 + PyYAML)" >&2
      exit 1
    fi
    PATCH="${RENDERED}"
  fi
  # Check "already applied" (reverse) FIRST: additive patches (pure insertions)
  # always pass a forward --check even when already present, so a forward-first
  # order would double-apply them and duplicate the inserted code.
  if git apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "${name} already applied"
  elif git apply --check "${PATCH}" 2>/dev/null; then
    git apply "${PATCH}"
    echo "Applied ${name}"
  else
    echo "${name} does not apply cleanly — resolve upstream drift manually" >&2
    exit 1
  fi
done
