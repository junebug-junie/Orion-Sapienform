#!/usr/bin/env bash
# Idempotent host setup: pyserial, udev, systemd unit for cabinet sensor reader.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/orion_runtime_root.sh
source "$SCRIPT_DIR/lib/orion_runtime_root.sh"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNTIME_ROOT="$(orion_resolve_runtime_root "$REPO_ROOT")"
VENV_PYTHON="$(orion_resolve_runtime_python "$REPO_ROOT")"
UDEV_SRC="$REPO_ROOT/deploy/udev/99-orion-cabinet-nano.rules"
UNIT_SRC="$REPO_ROOT/deploy/systemd/orion-cabinet-sensors.service"
UDEV_DST="/etc/udev/rules.d/99-orion-cabinet-nano.rules"
UNIT_DST="/etc/systemd/system/orion-cabinet-sensors.service"
DEFAULT_ENV="/etc/default/orion-cabinet-sensors"
SERVICE_USER="${ORION_CABINET_SERVICE_USER:-athena}"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: expected venv python at $VENV_PYTHON (runtime root: $RUNTIME_ROOT)" >&2
    exit 1
fi

if [[ "$REPO_ROOT" != "$RUNTIME_ROOT" ]]; then
    echo "note: invoked from worktree $REPO_ROOT; runtime/systemd will use $RUNTIME_ROOT" >&2
fi

echo "Installing pyserial into runtime venv ($VENV_PYTHON)..."
"$VENV_PYTHON" -m pip install -q pyserial

if [[ "$(id -u)" -ne 0 ]]; then
    echo "error: run as root (sudo) to install udev/systemd files" >&2
    exit 1
fi

if [[ ! -f "$UDEV_SRC" ]] || [[ ! -f "$UNIT_SRC" ]]; then
    echo "error: missing deploy artifacts under $REPO_ROOT/deploy/" >&2
    exit 1
fi

install -m 0644 "$UDEV_SRC" "$UDEV_DST"
echo "Installed $UDEV_DST"

# Materialize systemd unit with this checkout's absolute paths.
tmp_unit="$(mktemp)"
sed "s|@ORION_ROOT@|${RUNTIME_ROOT}|g" "$UNIT_SRC" >"$tmp_unit"
install -m 0644 "$tmp_unit" "$UNIT_DST"
rm -f "$tmp_unit"
echo "Installed $UNIT_DST (ORION_ROOT=$RUNTIME_ROOT)"

if [[ ! -f "$DEFAULT_ENV" ]]; then
    cat >"$DEFAULT_ENV" <<EOF
# Orion cabinet sensor reader — override service user if needed.
ORION_CABINET_SERVICE_USER=$SERVICE_USER
EOF
    chmod 0644 "$DEFAULT_ENV"
    echo "Created $DEFAULT_ENV"
else
    echo "$DEFAULT_ENV already exists (left unchanged)"
fi

if ! getent group plugdev >/dev/null 2>&1; then
    groupadd plugdev
    echo "Created group plugdev"
fi

if id "$SERVICE_USER" >/dev/null 2>&1; then
    usermod -aG plugdev,dialout "$SERVICE_USER" 2>/dev/null || usermod -aG plugdev "$SERVICE_USER"
    echo "Ensured $SERVICE_USER is in plugdev (and dialout when present)"
else
    echo "warning: user $SERVICE_USER not found; adjust User= in $UNIT_DST" >&2
fi

udevadm control --reload-rules
udevadm trigger --subsystem-match=tty
udevadm trigger --subsystem-match=usb --attr-match=idVendor=2341
echo "Reloaded udev rules (tty + Arduino USB/DFU)"


systemctl daemon-reload
systemctl enable orion-cabinet-sensors.service

OVERRIDE_DIR="/etc/systemd/system/orion-cabinet-sensors.service.d"
mkdir -p "$OVERRIDE_DIR"
cat >"$OVERRIDE_DIR/user.conf" <<EOF
[Service]
User=$SERVICE_USER
EOF
echo "Wrote $OVERRIDE_DIR/user.conf (User=$SERVICE_USER)"

systemctl daemon-reload
systemctl restart orion-cabinet-sensors.service
echo "Enabled and restarted orion-cabinet-sensors.service"

systemctl --no-pager --full status orion-cabinet-sensors.service || true
