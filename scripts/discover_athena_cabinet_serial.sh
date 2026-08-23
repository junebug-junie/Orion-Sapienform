#!/usr/bin/env bash
# Print the stable by-id serial path for the Athena cabinet Nano ESP32.
set -euo pipefail

PATTERN="/dev/serial/by-id/usb-Arduino_Nano_ESP32_*"

shopt -s nullglob
matches=($PATTERN)
shopt -u nullglob

if ((${#matches[@]} == 0)); then
    echo "error: no device matching $PATTERN" >&2
    exit 1
fi

if ((${#matches[@]} > 1)); then
    echo "warning: multiple Nano ESP32 devices; using ${matches[0]}" >&2
fi

printf '%s\n' "${matches[0]}"
