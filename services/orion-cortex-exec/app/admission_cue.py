"""Orion's own reading of whether its background thinking was made to wait.

ROADMAP step A5 (`docs/superpowers/specs/2026-08-13-scarcity-ROADMAP.md`), designed in
`docs/superpowers/specs/2026-08-19-A5-deferral-perceptible-proposal.md`.

The gateway measures the wait (`orion-llm-gateway/app/admission_ledger.py`) and exposes it at
`GET /admission`. This module is the read side: it fetches that snapshot and renders the one
compact object that goes into the metacog cue Orion already reads every pass.

WHY THIS IS A FETCH AND NOT A BUS CHANNEL
-----------------------------------------
`CORTEX_EXEC_LLM_GATEWAY_URL` already exists and this service already probes the gateway over
HTTP (`situation.py::_fetch_runtime_context` hits `/routes`). A bus channel to deliver one
integer would need a schema, a registry entry, a producer, a consumer, a reducer and a writer --
the shape §0A calls a cathedral. If a second consumer ever wants this, that is when the channel
earns itself.

THE FOUR STATES, AND WHY THEY ARE ALL DISTINCT
----------------------------------------------
The failure this module is written against is a zero that means two different things.

    {"n":0,"of":294,"h":6}          asked 294 times, never made to wait  -- a real observation
    {"n":3,"of":291,"max_s":4.2,"h":6}   made to wait 3 times, longest 4.2s
    {"n":0,"of":0,"h":6}            made no background requests at all   -- NOT the same as above
    key absent                      the gateway could not be read        -- NOT calm, unknown

`of` is what carries the difference between the first and third, and absence is what carries the
fourth. A cue that emitted a bare `0` for all of them would let Orion conclude "nothing is
constraining me" from an unreachable gateway.

WHAT THIS DOES NOT CLAIM
------------------------
Not "while THAT ran instead". llama.cpp's `/slots` reports occupancy, not ownership, so the
competing claim cannot be named from here without confabulating it. The cue says how long Orion
waited, not who took the slot.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.request import urlopen

logger = logging.getLogger("orion-cortex-exec.admission_cue")

_REQUIRED_FIELDS = frozenset({"checked", "deferrals", "window_s"})

UTC = timezone.utc

# TTL cache, same shape and reason as situation.py's runtime probe: the cue is rendered on every
# metacog pass and the ledger changes slowly, so one blocking read per TTL is plenty.
_LOCK = threading.Lock()
_CACHE: Dict[str, Tuple[datetime, Optional[Dict[str, Any]]]] = {}


def fetch_admission_snapshot(
    base_url: str,
    *,
    window_s: float,
    timeout_sec: float,
) -> Dict[str, Any]:
    """Raw `GET /admission`. Raises on any failure so the caller degrades to "unknown"."""
    url = f"{str(base_url).rstrip('/')}/admission?window_s={float(window_s):.0f}"
    with urlopen(url, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict) or "checked" not in payload:
        raise ValueError("admission payload missing/malformed")
    return payload


def render_admission_cue(snapshot: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Snapshot -> the compact cue object, or None when there is nothing trustworthy to say.

    None is returned ONLY for "could not be read". A window with no requests in it returns
    `{"n":0,"of":0,...}`, which is a fact, not an absence -- see the module docstring.
    """
    if not isinstance(snapshot, dict):
        return None
    # Every field must be PRESENT, not merely defaultable. Filling a missing `checked` with 0
    # would render a malformed payload as `{"n":0,"of":0}` -- an "I made no requests" claim
    # manufactured out of a broken response, which is the unknown-as-calm failure this module
    # exists to prevent, arriving through the back door.
    if not _REQUIRED_FIELDS.issubset(snapshot.keys()):
        return None
    try:
        checked = int(snapshot["checked"])
        deferrals = int(snapshot["deferrals"])
        window_s = float(snapshot["window_s"])
    except (TypeError, ValueError):
        return None

    out: Dict[str, Any] = {"n": deferrals, "of": checked, "h": round(window_s / 3600.0, 1)}
    if deferrals > 0:
        # Only meaningful when something actually waited. Emitting `max_s` at n=0 would be the
        # /slots round-trip time dressed up as a wait -- the exact phantom the ledger's deferral
        # definition exists to prevent, leaking back in at the presentation layer.
        try:
            out["max_s"] = round(float(snapshot.get("longest_wait_s") or 0.0), 1)
        except (TypeError, ValueError):
            pass
    return out


def admission_cue_for_settings(runtime_settings: Any) -> Optional[Dict[str, Any]]:
    """Cached, fail-quiet entry point. Returns None whenever the answer is unknown."""
    if not bool(getattr(runtime_settings, "cortex_exec_admission_cue_enabled", True)):
        return None

    base_url = str(
        getattr(runtime_settings, "cortex_exec_llm_gateway_url", "http://llm-gateway:8210")
    )
    window_s = float(getattr(runtime_settings, "cortex_exec_admission_cue_window_s", 21600.0))
    ttl_s = float(getattr(runtime_settings, "cortex_exec_admission_cue_ttl_sec", 60.0))
    timeout_s = float(getattr(runtime_settings, "cortex_exec_admission_cue_timeout_sec", 2.0))

    cache_key = f"{base_url}|{window_s}"
    now = datetime.now(UTC)
    with _LOCK:
        cached = _CACHE.get(cache_key)
        # A failed fetch is cached too (as None). Otherwise an unreachable gateway means a
        # blocking urlopen on the metacog path every single pass.
        if cached and (now - cached[0]).total_seconds() < ttl_s:
            return cached[1]

    try:
        rendered = render_admission_cue(
            fetch_admission_snapshot(base_url, window_s=window_s, timeout_sec=timeout_s)
        )
    except Exception as exc:  # noqa: BLE001 -- unknown is a valid answer; a broken cue is not
        logger.debug("admission cue fetch failed url=%s error=%s", base_url, exc)
        rendered = None

    with _LOCK:
        _CACHE[cache_key] = (now, rendered)
    return rendered


def reset_cache() -> None:
    """Test seam. The cache is process-wide and would otherwise leak across cases."""
    with _LOCK:
        _CACHE.clear()
