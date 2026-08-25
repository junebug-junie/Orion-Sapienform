#!/usr/bin/env bash
# Idempotent host setup: alsa-utils, audio group, systemd unit for ambient reader.
# Requires sudo — installs packages, systemd unit, and enables the service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/orion_runtime_root.sh
source "$SCRIPT_DIR/lib/orion_runtime_root.sh"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNTIME_ROOT="$(orion_resolve_runtime_root "$REPO_ROOT")"
VENV_PYTHON="$(orion_resolve_runtime_python "$REPO_ROOT")"
UNIT_SRC="$REPO_ROOT/deploy/systemd/orion-ambient-audio.service"
UNIT_DST="/etc/systemd/system/orion-ambient-audio.service"
DEFAULT_ENV="/etc/default/orion-ambient-audio"
SERVICE_USER="${ORION_AMBIENT_SERVICE_USER:-athena}"
DEFAULT_DEVICE="${ORION_AMBIENT_AUDIO_DEVICE:-plughw:CARD=CMTECK,DEV=0}"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: expected venv python at $VENV_PYTHON (runtime root: $RUNTIME_ROOT)" >&2
    exit 1
fi

if [[ "$REPO_ROOT" != "$RUNTIME_ROOT" ]]; then
    echo "note: invoked from worktree $REPO_ROOT; runtime/systemd will use $RUNTIME_ROOT" >&2
fi

if [[ "$(id -u)" -ne 0 ]]; then
    echo "error: run as root (sudo) to install packages, systemd unit, and group membership" >&2
    echo "usage: sudo $0" >&2
    exit 1
fi

if ! command -v arecord >/dev/null 2>&1; then
    echo "Installing alsa-utils (provides arecord)..."
    apt-get update -qq
    apt-get install -y alsa-utils
else
    echo "alsa-utils already installed ($(arecord --version 2>&1 | head -1))"
fi

if [[ ! -f "$UNIT_SRC" ]]; then
    echo "error: missing deploy artifact $UNIT_SRC" >&2
    exit 1
fi

tmp_unit="$(mktemp)"
sed "s|@ORION_ROOT@|${RUNTIME_ROOT}|g" "$UNIT_SRC" >"$tmp_unit"
install -m 0644 "$tmp_unit" "$UNIT_DST"
rm -f "$tmp_unit"
echo "Installed $UNIT_DST (ORION_ROOT=$RUNTIME_ROOT)"

if [[ ! -f "$DEFAULT_ENV" ]]; then
    cat >"$DEFAULT_ENV" <<EOF
# Orion ambient audio reader — override service user or ALSA device if needed.
ORION_AMBIENT_SERVICE_USER=$SERVICE_USER
ORION_AMBIENT_AUDIO_DEVICE=$DEFAULT_DEVICE
EOF
    chmod 0644 "$DEFAULT_ENV"
    echo "Created $DEFAULT_ENV"
else
    echo "$DEFAULT_ENV already exists (left unchanged)"
fi

if id "$SERVICE_USER" >/dev/null 2>&1; then
    usermod -aG audio "$SERVICE_USER"
    echo "Ensured $SERVICE_USER is in group audio"
else
    echo "warning: user $SERVICE_USER not found; adjust User= in $UNIT_DST" >&2
fi

systemctl daemon-reload
systemctl enable orion-ambient-audio.service

OVERRIDE_DIR="/etc/systemd/system/orion-ambient-audio.service.d"
mkdir -p "$OVERRIDE_DIR"
cat >"$OVERRIDE_DIR/user.conf" <<EOF
[Service]
User=$SERVICE_USER
EOF
echo "Wrote $OVERRIDE_DIR/user.conf (User=$SERVICE_USER)"

systemctl daemon-reload
systemctl restart orion-ambient-audio.service
echo "Enabled and restarted orion-ambient-audio.service"

echo "Verifying ALSA device discovery..."
export ORION_AMBIENT_AUDIO_DEVICE="$DEFAULT_DEVICE"
if ! "$SCRIPT_DIR/discover_athena_ambient_audio.sh"; then
    echo "warning: discover script failed; check CMTECK mic is attached" >&2
fi

echo "One-shot capture verify (as $SERVICE_USER)..."
if sudo -u "$SERVICE_USER" arecord -D "$DEFAULT_DEVICE" -f S16_LE -r 16000 -c 1 -t raw -d 1 -q /dev/null 2>/dev/null; then
    echo "ok: one-shot arecord capture succeeded"
else
    echo "warning: one-shot capture failed; user may need to re-login for audio group" >&2
fi

systemctl --no-pager --full status orion-ambient-audio.service || true
