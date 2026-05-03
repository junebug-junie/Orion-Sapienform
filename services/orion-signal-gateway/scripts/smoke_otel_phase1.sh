#!/usr/bin/env bash
# Phase 1 acceptance helpers (spec §8.1): verify Tempo + Grafana from compose are reachable.
# Does not emit OTLP spans (run gateway traffic separately). Exit 0 when endpoints respond.

set -euo pipefail

TEMPO_HTTP="${TEMPO_HTTP_URL:-http://127.0.0.1:${TEMPO_HTTP_PORT:-3200}}"
GRAFANA_HTTP="${GRAFANA_HTTP_URL:-http://127.0.0.1:${GRAFANA_HTTP_PORT:-3001}}"
COLLECTOR_METRICS="${COLLECTOR_METRICS_URL:-http://127.0.0.1:8889/metrics}"

echo "smoke_otel_phase1: tempo ready ${TEMPO_HTTP}"
curl -sfS "${TEMPO_HTTP}/ready" >/dev/null

echo "smoke_otel_phase1: grafana root ${GRAFANA_HTTP}"
curl -sfS -o /dev/null "${GRAFANA_HTTP}/"

echo "smoke_otel_phase1: collector prometheus scrape ${COLLECTOR_METRICS}"
out="$(curl -sfS --max-time 5 "${COLLECTOR_METRICS}")"
if ! printf '%s' "$out" | grep -q .; then
  echo "smoke_otel_phase1: empty metrics response" >&2
  exit 1
fi

echo "smoke_otel_phase1: ok"
