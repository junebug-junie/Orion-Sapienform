#!/usr/bin/env bash
# Flash Athena cabinet sensor firmware to Arduino Nano ESP32.
#
# Discovers the stable by-id serial path (never ttyACM* directly).
# Uses arduino-cli when available; prints clear install instructions otherwise.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKETCH_DIR="${REPO_ROOT}/firmware/athena-cabinet-nano"
FQBN="arduino:esp32:nano_nora"
BAUD="115200"

COMPILE_ONLY=0
UPLOAD=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [--compile-only] [--upload]

  --compile-only   Compile the sketch; do not upload (default when no device).
  --upload         Compile and upload (default when a Nano ESP32 by-id path exists).

Environment:
  ARDUINO_CLI       Path to arduino-cli (default: arduino-cli on PATH)
  CABINET_NANO_FQBN Override board FQBN (default: ${FQBN})

Requires arduino-cli with board ${FQBN} and libraries listed in
firmware/athena-cabinet-nano/README.md.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile-only)
      COMPILE_ONLY=1
      UPLOAD=0
      shift
      ;;
    --upload)
      COMPILE_ONLY=0
      UPLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${SKETCH_DIR}" ]]; then
  echo "ERROR: sketch directory not found: ${SKETCH_DIR}" >&2
  exit 1
fi

if [[ ! -f "${SKETCH_DIR}/athena-cabinet-nano.ino" ]]; then
  echo "ERROR: sketch file not found: ${SKETCH_DIR}/athena-cabinet-nano.ino" >&2
  exit 1
fi

ARDUINO_CLI="${ARDUINO_CLI:-arduino-cli}"

if ! command -v "${ARDUINO_CLI}" >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: arduino-cli not found on PATH.

Install arduino-cli: https://arduino.github.io/arduino-cli/latest/installation/
Then install the board core and libraries:

  arduino-cli core update-index
  arduino-cli core install arduino:esp32
  arduino-cli lib install "Adafruit BME680 Library"
  arduino-cli lib install "Adafruit LTR390 Library"
  arduino-cli lib install "Adafruit LIS3MDL"
  arduino-cli lib install "Adafruit VL53L1X"
  arduino-cli lib install "Adafruit BNO08x"
  arduino-cli lib install "Adafruit BNO08x RVC"
  arduino-cli lib install "Adafruit PM25 AQI Sensor"
  arduino-cli lib install "ArduinoJson"

Compile manually:

  ${ARDUINO_CLI} compile --fqbn ${FQBN} ${SKETCH_DIR}
EOF
  exit 1
fi

FQBN="${CABINET_NANO_FQBN:-${FQBN}}"

discover_device() {
  local -a matches=()
  shopt -s nullglob
  matches=(/dev/serial/by-id/usb-Arduino_Nano_ESP32_*)
  shopt -u nullglob

  if [[ ${#matches[@]} -eq 0 ]]; then
    return 1
  fi
  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "WARN: multiple Nano ESP32 by-id devices; using first: ${matches[0]}" >&2
  fi
  printf '%s' "${matches[0]}"
}

DEVICE=""
if DEVICE="$(discover_device)"; then
  echo "Discovered device: ${DEVICE}"
else
  echo "WARN: no /dev/serial/by-id/usb-Arduino_Nano_ESP32_* device found." >&2
  if [[ "${UPLOAD}" -eq 1 && "${COMPILE_ONLY}" -eq 0 ]]; then
    echo "Falling back to compile-only." >&2
    COMPILE_ONLY=1
    UPLOAD=0
  fi
fi

echo "Compiling ${SKETCH_DIR} for ${FQBN}..."
"${ARDUINO_CLI}" compile --fqbn "${FQBN}" "${SKETCH_DIR}"

if [[ "${COMPILE_ONLY}" -eq 1 ]]; then
  echo "Compile succeeded (upload skipped)."
  exit 0
fi

if [[ -z "${DEVICE}" ]]; then
  echo "ERROR: upload requested but no device discovered." >&2
  exit 1
fi

if systemctl is-active --quiet orion-cabinet-sensors.service 2>/dev/null; then
  echo "WARN: orion-cabinet-sensors.service is active; stop it before upload:" >&2
  echo "  sudo systemctl stop orion-cabinet-sensors.service" >&2
fi

echo "Uploading to ${DEVICE}..."
if ! "${ARDUINO_CLI}" upload -p "${DEVICE}" --fqbn "${FQBN}" "${SKETCH_DIR}"; then
  cat >&2 <<EOF
ERROR: upload failed (often LIBUSB_ERROR_ACCESS until DFU udev is installed).

Re-run host setup from this repo, then flash again:
  sudo scripts/setup_athena_cabinet_sensors.sh
  sudo systemctl stop orion-cabinet-sensors.service
  ./scripts/flash_athena_cabinet_nano.sh --upload

If it still fails, unplug/replug the Nano once so udev re-applies.
EOF
  exit 1
fi

echo "Upload complete. Serial monitor at ${BAUD} baud:"
echo "  ${ARDUINO_CLI} monitor -p ${DEVICE} -c baudrate=${BAUD}"
