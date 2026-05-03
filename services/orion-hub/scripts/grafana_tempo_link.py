"""Grafana Explore deep links for Tempo trace-id search (spec Phase 1).

Explore URLs include ``orgId`` (default from ``HUB_OTEL_GRAFANA_ORG_ID``) for
single-org setups; set the env when the Tempo datasource lives in another org.
"""
from __future__ import annotations

import json
from urllib.parse import quote

from .otel_trace_id import is_valid_otel_trace_id, normalize_otel_trace_id

GRAFANA_EXPLORE_DEFAULT_ORG_ID = 1


def build_grafana_tempo_trace_explore_url(
    *,
    grafana_base_url: str,
    trace_id: str,
    datasource_uid: str = "tempo",
    grafana_org_id: int = GRAFANA_EXPLORE_DEFAULT_ORG_ID,
) -> str | None:
    """Return Explore URL for Tempo trace-by-id, or None if trace id is invalid."""
    tid = normalize_otel_trace_id(trace_id)
    if not is_valid_otel_trace_id(tid):
        return None
    base = grafana_base_url.strip().rstrip("/")
    if not base:
        return None
    left = {
        "datasource": datasource_uid,
        "queries": [{"query": tid, "queryType": "traceId"}],
        "range": {"from": "now-15m", "to": "now"},
    }
    encoded = quote(json.dumps(left, separators=(",", ":")))
    return f"{base}/explore?orgId={int(grafana_org_id)}&left={encoded}"
