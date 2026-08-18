"""Producer-side update for `orion.field.significance`'s level-aware
sustained-load reading.

See `orion/schemas/field_state.py`'s `sustained_load_pressure`/
`sustained_load_computed_at` docstrings and `docs/superpowers/specs/2026-08-
16-level-aware-significance-design.md` for the full account.

WHY THROTTLED, INLINE, NOT A SEPARATE WORKER LOOP
---------------------------------------------------
`substrate_field_state` is append-only tick history, not a single mutable
"current state" row -- every hot tick's `save_field()`/`commit_digest_tick()`
call is a fresh `INSERT` with a new `tick_id` (see `app/store.py::save_field`).
A separate slow-cadence loop with its own `save_field()` call would either
insert its own near-duplicate near-empty tick (wasteful, and confuses any
consumer assuming each row is one coherent digestion snapshot) or, if it
instead loaded and re-saved the row that was "latest" when it started, could
lose a race against the hot loop inserting a newer one in the meantime and
silently amend a row that is no longer current. Neither is needed: this
function is called from the SAME hot tick's SAME `state` object, right
alongside `update_tension_pressure`, so there is exactly one writer and
exactly one `save_field()` call, same as every other digestion step. The only
difference from `update_tension_pressure` is that THIS read is a real
Postgres query (not an O(1) EWMA update), so it is throttled internally
rather than run on every ~2s tick.
"""
from __future__ import annotations

import logging

from orion.field.significance import compute_tick, sustained_load_pressure
from orion.schemas.field_state import FieldStateV1

logger = logging.getLogger("orion.field.digester.significance")


def update_significance_pressure(
    state: FieldStateV1,
    store,
    *,
    window_seconds: float,
    check_interval_sec: float,
) -> FieldStateV1:
    """Recompute `state.sustained_load_pressure` if `check_interval_sec` has
    elapsed since it was last actually recomputed, else leave it carried
    forward unchanged from the prior tick -- same "a value nothing refreshed
    this tick keeps its last real reading" convention decayed/held channels
    already use elsewhere in this pipeline, just without any decay (nothing
    here trends toward a floor between refreshes; it simply is not
    re-measured yet).

    Never raises: a DB failure during the window read degrades to "leave the
    existing value in place, try again next tick" -- explicitly does NOT
    advance `sustained_load_computed_at` on failure, so a transient DB hiccup
    causes a retry on the very next hot tick rather than a stale value
    sitting untouched for a full `check_interval_sec`.
    """
    last = state.sustained_load_computed_at
    now = state.generated_at
    if last is not None and (now - last).total_seconds() < check_interval_sec:
        return state

    try:
        payloads = store.load_recent_field_json(window_seconds=window_seconds)
        tick = compute_tick(payloads, window_seconds=window_seconds)
    except Exception:  # noqa: BLE001 - a broken read must not crash the digestion tick
        logger.warning("significance_pressure_update_failed", exc_info=True)
        return state

    state.sustained_load_pressure = sustained_load_pressure(tick)
    state.sustained_load_computed_at = now
    return state
