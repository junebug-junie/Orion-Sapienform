#!/usr/bin/env bash
# Print the stable plughw ALSA device string for the Athena CMTECK USB mic.
set -euo pipefail

CARD_NAME="${ORION_AMBIENT_AUDIO_CARD:-CMTECK}"
DEV="${ORION_AMBIENT_AUDIO_DEV:-0}"
WAIT_SEC="${ORION_AMBIENT_AUDIO_DISCOVER_WAIT_SEC:-0}"

find_card() {
    CARD_LINE=""
    if command -v arecord >/dev/null 2>&1; then
        CARD_LINE="$(arecord -l 2>/dev/null | grep -i "$CARD_NAME" | head -1 || true)"
    fi
    if [[ -z "$CARD_LINE" ]] && command -v aplay >/dev/null 2>&1; then
        CARD_LINE="$(aplay -l 2>/dev/null | grep -i "$CARD_NAME" | head -1 || true)"
    fi
    printf '%s' "$CARD_LINE"
}

card_line="$(find_card)"

if [[ -z "$card_line" ]] && [[ "$WAIT_SEC" != "0" ]]; then
    deadline=$((SECONDS + WAIT_SEC))
    while [[ -z "$card_line" ]] && ((SECONDS < deadline)); do
        sleep 0.5
        card_line="$(find_card)"
    done
fi

if [[ -z "$card_line" ]]; then
    echo "error: no ALSA capture card matching name '$CARD_NAME'" >&2
    echo "hint: run 'arecord -l' and set ORION_AMBIENT_AUDIO_CARD if the mic uses a different name" >&2
    exit 1
fi

device_override="${ORION_AMBIENT_AUDIO_DEVICE:-}"
if [[ -n "$device_override" ]]; then
    printf '%s\n' "$device_override"
    exit 0
fi

printf 'plughw:CARD=%s,DEV=%s\n' "$CARD_NAME" "$DEV"
