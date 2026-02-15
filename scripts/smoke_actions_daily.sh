#!/usr/bin/env bash
set -euo pipefail

export ORION_BUS_URL="${ORION_BUS_URL:-redis://localhost:6379/0}"
export NOTIFY_BASE_URL="${NOTIFY_BASE_URL:-http://localhost:7140}"
export NOTIFY_API_TOKEN="${NOTIFY_API_TOKEN:-}"
export ACTIONS_DAILY_RUN_ONCE_DATE="${ACTIONS_DAILY_RUN_ONCE_DATE:-}"

python scripts/smoke_daily_actions.py --action pulse --timeout "${SMOKE_TIMEOUT:-90}"
python scripts/smoke_daily_actions.py --action metacog --timeout "${SMOKE_TIMEOUT:-90}"
