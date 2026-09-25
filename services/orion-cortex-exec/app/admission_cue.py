"""Orion's own reading of whether its background thinking was made to wait.

ROADMAP step A5 (`docs/superpowers/specs/2026-08-13-scarcity-ROADMAP.md`), designed in
`docs/superpowers/specs/2026-08-19-A5-deferral-perceptible-proposal.md`.

Since 2026-09-24 every LLM call waits in orion-gpu-pool, which records each lease's wait in
`gpu_pool_events` (via sql-writer). This module is the read side: one windowed SQL read over the
POSTGRES_URI this process already has, rendered into the one compact object that goes into the
metacog cue Orion already reads every pass. (It used to read the LLM gateway's in-process ledger
at `GET /admission`; that ledger was deleted when the gateway cut over to the pool.)

THE FOUR STATES, AND WHY THEY ARE ALL DISTINCT
----------------------------------------------
The failure this module is written against is a zero that means two different things.

    {"n":0,"of":294,"h":6}          asked 294 times, never made to wait  -- a real observation
    {"n":3,"of":291,"max_s":4.2,"h":6}   made to wait 3 times, longest 4.2s
    {"n":0,"of":0,"h":6}            made no background requests at all   -- NOT the same as above
    key absent                      the pool history could not be read   -- NOT calm, unknown

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

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("orion-cortex-exec.admission_cue")

_REQUIRED_FIELDS = frozenset({"checked", "deferrals", "window_s", "unchecked"})

UTC = timezone.utc

# TTL cache, same shape and reason as situation.py's runtime probe: the cue is rendered on every
# metacog pass and the ledger changes slowly, so one blocking read per TTL is plenty.
_LOCK = threading.Lock()
_CACHE: Dict[str, Tuple[datetime, Optional[Dict[str, Any]]]] = {}


# A background lease that waited at least this long was "made to wait". A grant that arrives in the
# same few milliseconds as the request is bookkeeping, not waiting -- counting it would be the
# phantom-wait this cue was built to refuse (see render_admission_cue's max_s note).
WAIT_THRESHOLD_MS = 500.0

_ENGINE: Any = None


def _engine() -> Any:
    global _ENGINE
    if _ENGINE is None:
        import os

        from sqlalchemy import create_engine

        dsn = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError("POSTGRES_URI not set")
        _ENGINE = create_engine(dsn, pool_pre_ping=True, pool_size=1, max_overflow=1)
    return _ENGINE


def fetch_admission_snapshot(*, window_s: float, engine: Any = None) -> Dict[str, Any]:
    """Orion's own background leases in orion-gpu-pool over the window (gpu_pool_events).

    Replaces the LLM gateway's in-process ledger (``GET /admission``), deleted when the gateway cut
    over to the pool (2026-09-24). Same snapshot keys, so ``render_admission_cue`` is unchanged:
      checked   -- background leases Orion asked for (admitted events)
      deferrals -- of those, granted only after waiting >= WAIT_THRESHOLD_MS, or never served
      longest_wait_s -- the longest such wait
      unchecked -- always 0: the pool measures every wait, unlike /slots polling that could fail open
    First-person only: holders starting "http:" are other callers of the gateway's HTTP API (e.g.
    AI Town NPC dialogue over the OpenAI passthrough), not Orion. Raises on any failure so the
    caller degrades to "unknown".
    """
    from sqlalchemy import text

    sql = text(
        """
        SELECT
          count(*) FILTER (WHERE event = 'admitted') AS checked,
          count(*) FILTER (WHERE (event = 'granted' AND waited_ms >= :threshold)
                                OR event = 'unavailable') AS deferrals,
          max(waited_ms) FILTER (WHERE event = 'granted' AND waited_ms >= :threshold) AS longest_ms
        FROM gpu_pool_events
        WHERE priority = 'background'
          AND holder NOT LIKE 'http:%'
          AND generated_at >= now() - (:window_s * interval '1 second')
        """
    )
    with (engine or _engine()).connect() as conn:
        row = conn.execute(sql, {"threshold": WAIT_THRESHOLD_MS, "window_s": float(window_s)}).mappings().one()
    return {
        "checked": int(row["checked"] or 0),
        "deferrals": int(row["deferrals"] or 0),
        "unchecked": 0,
        "window_s": float(window_s),
        "longest_wait_s": float(row["longest_ms"] or 0.0) / 1000.0,
    }


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
        unchecked = int(snapshot["unchecked"])
        window_s = float(snapshot["window_s"])
    except (TypeError, ValueError):
        return None

    # The gate FAILS OPEN: when llama.cpp's /slots is unreachable or disabled, the request is
    # forwarded without checking and recorded as `unchecked`. Those requests count toward
    # `checked` but can never be deferrals, so a window where /slots was down throughout holds
    # `{checked: 294, deferrals: 0, unchecked: 294}` -- and rendering that as "asked 294 times,
    # never constrained" would be a confident first-person claim assembled entirely from 294
    # measurements that did not happen. If nothing in the window was measurable, there is
    # nothing trustworthy to say.
    if checked > 0 and unchecked >= checked:
        return None

    out: Dict[str, Any] = {"n": deferrals, "of": checked, "h": round(window_s / 3600.0, 1)}
    if unchecked > 0:
        # Partially measurable: report the count so the denominator is not read as fully
        # observed.
        out["unk"] = unchecked
    if deferrals > 0:
        # Only meaningful when something actually waited. Emitting `max_s` at n=0 would be the
        # /slots round-trip time dressed up as a wait -- the exact phantom the ledger's deferral
        # definition exists to prevent, leaking back in at the presentation layer.
        try:
            longest = float(snapshot.get("longest_wait_s") or 0.0)
        except (TypeError, ValueError):
            longest = 0.0
        if longest > 0.0:
            # Sub-0.05s waits round to 0.0 at one decimal, which would render "I was made to
            # wait, for zero seconds" -- the same self-contradiction the comment above rejects.
            # Reachable with LLM_GATEWAY_BACKGROUND_MAX_WAIT_SEC=0, where the first busy poll
            # times out immediately. Keep more precision rather than rounding a real wait away.
            out["max_s"] = round(longest, 1) if longest >= 0.1 else round(longest, 3)
    return out


def admission_cue_for_settings(runtime_settings: Any) -> Optional[Dict[str, Any]]:
    """Cached, fail-quiet entry point. Returns None whenever the answer is unknown."""
    if not bool(getattr(runtime_settings, "cortex_exec_admission_cue_enabled", True)):
        return None

    window_s = float(getattr(runtime_settings, "cortex_exec_admission_cue_window_s", 21600.0))
    ttl_s = float(getattr(runtime_settings, "cortex_exec_admission_cue_ttl_sec", 60.0))

    cache_key = f"gpu_pool|{window_s}"
    now = datetime.now(UTC)
    with _LOCK:
        cached = _CACHE.get(cache_key)
        # A failed read is cached too (as None). Otherwise an unreachable database means a
        # blocking query on the metacog path every single pass.
        if cached and (now - cached[0]).total_seconds() < ttl_s:
            return cached[1]

    try:
        rendered = render_admission_cue(fetch_admission_snapshot(window_s=window_s))
    except Exception as exc:  # noqa: BLE001 -- unknown is a valid answer; a broken cue is not
        logger.debug("admission cue read failed error=%s", exc)
        rendered = None

    with _LOCK:
        _CACHE[cache_key] = (now, rendered)
    return rendered


def reset_cache() -> None:
    """Test seam. The cache is process-wide and would otherwise leak across cases."""
    with _LOCK:
        _CACHE.clear()
