#!/bin/sh
set -e

if [ -z "$ORION_BUS_URL" ]; then
  echo "❌ ORION_BUS_URL is not set. Set it to something like: redis://100.92.216.81:6379/0"
  exit 1
fi

BUS_HOST="$(echo "$ORION_BUS_URL" | sed -E 's#redis://([^:/]+).*#\1#')"
BUS_PORT="$(echo "$ORION_BUS_URL" | sed -E 's#redis://[^:/]+:([0-9]+).*#\1#')"
[ -z "$BUS_PORT" ] && BUS_PORT=6379

echo "🕓 Waiting for Redis at $BUS_HOST:$BUS_PORT ..."

while :; do
  if command -v redis-cli >/dev/null 2>&1; then
    # Fastest + most reliable: actual PING to the URL
    if redis-cli -u "$ORION_BUS_URL" ping 2>/dev/null | grep -q PONG; then
      echo "✅ Redis responded to PING at $ORION_BUS_URL"
      break
    fi
  elif command -v nc >/dev/null 2>&1; then
    # Fallback: TCP port check
    if nc -z "$BUS_HOST" "$BUS_PORT" >/dev/null 2>&1; then
      echo "✅ TCP port open at $BUS_HOST:$BUS_PORT"
      break
    fi
  else
    # Last resort: ping host only
    if ping -c1 -W1 "$BUS_HOST" >/dev/null 2>&1; then
      echo "✅ Host reachable at $BUS_HOST (port not verified)"
      break
    fi
  fi

  echo "⏳ Still waiting for $BUS_HOST:$BUS_PORT ... retrying in 2s"
  sleep 2
done

echo "🚀 Starting Hub..."
exec "$@"
