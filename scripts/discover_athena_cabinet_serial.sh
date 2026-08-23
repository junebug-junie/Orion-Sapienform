#!/usr/bin/env bash
# Print the stable by-id serial path for the Athena cabinet Nano ESP32.
set -euo pipefail

PATTERN="/dev/serial/by-id/usb-Arduino_Nano_ESP32_*"
WAIT_SEC="${ORION_CABINET_DISCOVER_WAIT_SEC:-0}"

find_matches() {
    shopt -s nullglob
    matches=($PATTERN)
    shopt -u nullglob
}

find_matches

if ((${#matches[@]} == 0)) && [[ "$WAIT_SEC" != "0" ]]; then
    deadline=$((SECONDS + WAIT_SEC))
    while ((${#matches[@]} == 0)) && ((SECONDS < deadline)); do
        sleep 0.5
        find_matches
    done
fi

if ((${#matches[@]} == 0)); then
    echo "error: no device matching $PATTERN" >&2
    if [[ "$WAIT_SEC" == "0" ]]; then
        echo "hint: after DFU flash, wait for re-enumeration or set ORION_CABINET_DISCOVER_WAIT_SEC=30" >&2
    fi
    exit 1
fi

if ((${#matches[@]} > 1)); then
    echo "warning: multiple Nano ESP32 devices; using ${matches[0]}" >&2
fi

printf '%s\n' "${matches[0]}"
