"""Who still sends a lane, and which calls lane routing actually moves (GPU pool stage 6.4).

``LLM_LANE_ROUTING_ENABLED`` (app/lane_routes.py) is on the stage 6 delete list, but deleting it
changes routing for any call it currently re-routes. Before that PR, a 24 h window has to show
every such call is zero or understood (spec 2026-09-30-gpu-pool-stage6-telemetry-reducers-lockdown.md,
item 6.4). This module counts, per bus request planned by ``plan_llm_chat``:

- ``route_chosen``: the route the call actually leases for today (lane routing's pick, or
  ``rejected`` when lane routing refused it);
- ``route_without_lane_routing``: what ``_resolve_route`` alone would pick -- i.e. the route the
  call would get after the deletion PR.

``rerouted`` is the two differing: those are exactly the calls whose behaviour the deletion changes.
A call is recorded when it carries a lane (``options.llm_lane`` / ``options.execution_lane``) OR is
rerouted without one (lane routing's chat default ignores ``options.route``/``routing_key`` and turns
an unknown route into ``quick``). Calls with neither are only counted in ``requests_total``.

Two records, like routes_compat_reads: one INFO log line per recorded call (``llm_gateway_lane_sender``,
survives a restart in ``docker logs``) and an in-process counter since boot
(``GET /debug/lane-senders``). Deletion is safe on its own when ``rerouted_total == 0`` with
``uptime_sec >= 86400``; otherwise each ``by_sender`` row with ``rerouted: true`` needs an answer.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("orion-llm-gateway.lane_senders")

_MAX_ROWS = 64  # (+2 overflow rows) a caller cycling sources/lanes must not grow this without bound
_MAX_FIELD = 48

_lock = threading.Lock()
_state: Dict[str, Any] = {}


def reset() -> None:
    with _lock:
        _state.clear()
        _state.update(started_monotonic=time.monotonic(),
                      started_at=datetime.now(timezone.utc).isoformat(),
                      requests_total=0, lane_requests_total=0, rerouted_total=0,
                      rerouted_without_lane_total=0, last_recorded_at=None, by_sender={})


reset()


def _clean(value: Any, default: str = "-") -> str:
    s = str(value).strip() if value is not None else ""
    return (s or default)[:_MAX_FIELD]


def lane_field(options: Optional[Dict[str, Any]]) -> Optional[str]:
    """The lane exactly as sent, keyed by the field lane_routes.resolve_llm_lane_route reads
    (``options.get("llm_lane") or options.get("execution_lane")``, same truthiness). None when
    the call names no lane. Never raises: it runs on the request path."""
    try:
        opts = options if isinstance(options, dict) else {}
        for key in ("llm_lane", "execution_lane"):
            raw = opts.get(key)
            if raw:
                return f"{key}={_clean(raw).lower()}"
        return None
    except Exception:  # noqa: BLE001 -- instrumentation must never fail a call
        return None


def _is_rejected(route: str) -> bool:
    return route == "rejected" or route.startswith("rejected:")


def record(*, source: Optional[str], lane: Optional[str], route_in: Optional[str],
           route_chosen: Optional[str], route_without_lane_routing: Optional[str],
           lane_routing: str, corr: Optional[str] = None) -> None:
    """One planned call. ``lane_routing`` is ``applied`` / ``skipped_hold`` / ``disabled``.

    ``route_chosen`` None means lane routing refused the call; ``route_without_lane_routing``
    is passed as ``rejected:<name>`` by the caller when that name is not a pool route (the call
    would fail after the deletion, not move). Never raises: it runs on the request path."""
    try:
        _record(source=source, lane=lane, route_in=route_in, route_chosen=route_chosen,
                route_without_lane_routing=route_without_lane_routing,
                lane_routing=lane_routing, corr=corr)
    except Exception:  # noqa: BLE001 -- instrumentation must never fail a call
        logger.debug("lane_senders.record failed", exc_info=True)


def _record(*, source: Optional[str], lane: Optional[str], route_in: Optional[str],
            route_chosen: Optional[str], route_without_lane_routing: Optional[str],
            lane_routing: str, corr: Optional[str]) -> None:
    chosen_raw = "rejected" if route_chosen is None else str(route_chosen)
    without_raw = str(route_without_lane_routing or "-")
    # Compare untruncated; both sides failing is not a behaviour change worth an answer.
    rerouted = chosen_raw != without_raw and not (_is_rejected(chosen_raw) and _is_rejected(without_raw))
    chosen, without = _clean(chosen_raw), _clean(without_raw)
    with _lock:
        _state["requests_total"] += 1
        if lane is None and not rerouted:
            return
        now = datetime.now(timezone.utc).isoformat()
        if lane is not None:
            _state["lane_requests_total"] += 1
        if rerouted:
            _state["rerouted_total"] += 1
            if lane is None:
                _state["rerouted_without_lane_total"] += 1
        _state["last_recorded_at"] = now
        row_fields: Dict[str, Any] = {
            "source": _clean(source, "unknown"), "lane": lane or "none", "route_in": _clean(route_in),
            "route_chosen": chosen, "route_without_lane_routing": without,
            "lane_routing": lane_routing, "rerouted": rerouted}
        key = " ".join(f"{k}={v}" for k, v in row_fields.items())
        rows: Dict[str, Any] = _state["by_sender"]
        if key not in rows and len(rows) >= _MAX_ROWS:
            # Overflow keeps the rerouted split, so a reroute is never hidden in an anonymous row.
            key = f"other rerouted={rerouted}"
            row_fields = {"source": "other", "rerouted": rerouted}
        row = rows.setdefault(key, {**row_fields, "requests": 0, "last_at": None})
        row["requests"] += 1
        row["last_at"] = now
    logger.info("llm_gateway_lane_sender corr=%s source=%s lane=%s route_in=%s route_chosen=%s "
                "route_without_lane_routing=%s rerouted=%s lane_routing=%s",
                corr, _clean(source, "unknown"), lane or "none", _clean(route_in), chosen, without,
                rerouted, lane_routing)


def snapshot() -> Dict[str, Any]:
    with _lock:
        rows = sorted((dict(v) for v in _state["by_sender"].values()),
                      key=lambda r: (-r["requests"], r.get("source", "")))
        return {
            "counting_since": _state["started_at"],
            "uptime_sec": round(time.monotonic() - _state["started_monotonic"], 1),
            "requests_total": _state["requests_total"],
            "lane_requests_total": _state["lane_requests_total"],
            "rerouted_total": _state["rerouted_total"],
            "rerouted_without_lane_total": _state["rerouted_without_lane_total"],
            "last_recorded_at": _state["last_recorded_at"],
            "by_sender": rows,
        }
