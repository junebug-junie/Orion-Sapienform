"""Replay-only adapters for retired source kinds (reveries that ask, Patch 1, 2026-11-24).

``reverie_chain`` and ``expectation_verdict`` have no producer and no live reader any more
(``orion.temporal_self.sources.RETIRED_KINDS``). Frozen fixture days exported before the retirement
(``orion/temporal_self/evals/fixtures/day_2026-10-09.jsonl.gz``) still carry their rows, and the reducer
still folds them (``arcs.py``, history only), so replay evals read them through these adapters. Nothing in
a service imports this module: a live read can never emit a retired kind.
"""

from __future__ import annotations

from typing import Callable

from orion.schemas.temporal_self import TemporalSelfEventV1
from orion.temporal_self.day import DEFAULT_TZ, as_utc
from orion.temporal_self.sources import Row, _event, _iso, clip


def reverie_chain(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """substrate_reverie_chain plus its thoughts (``row['thoughts']``: thought_id, created_at,
    correlation_id; linked by thought_json.chain_id). Emitted only once ``terminal_reason`` is
    set. A chain with no thoughts has no content and is not an arc. Time: first thought .. max(last thought, chain created_at)."""
    if not row.get("terminal_reason"):
        return None  # still running: emitting now would freeze a partial chain into a closed arc
    thoughts = sorted(
        (t for t in (row.get("thoughts") or []) if t.get("thought_id") and t.get("created_at")),
        key=lambda t: (as_utc(t["created_at"]), str(t["thought_id"])),
    )
    if not thoughts:
        return None
    first = as_utc(thoughts[0]["created_at"])
    last = max(as_utc(thoughts[-1]["created_at"]), as_utc(row.get("created_at")) or first)
    return _event(
        kind="reverie_chain", table="substrate_reverie_chain", ref=row.get("chain_id"), at=first,
        ended=last, tz_name=tz_name, subject_ref=str(row.get("chain_id")),
        related_refs=sorted({str(t["correlation_id"]) for t in thoughts if t.get("correlation_id")}),
        label=clip(row.get("theme_key")),
        payload={"terminal_reason": row.get("terminal_reason"),
                 "thought_ids": [str(t["thought_id"]) for t in thoughts]},
    )


def expectation_verdict(row: Row, tz_name: str = DEFAULT_TZ) -> TemporalSelfEventV1 | None:
    """substrate_reverie_thought once scored. Time: expectation_scored_at. Attaches to its
    reverie arc by chain id (``row['chain_id']`` = thought_json.chain_id)."""
    if row.get("expectation_scored_at") is None or not row.get("expectation_verdict"):
        return None
    chain = row.get("chain_id")
    return _event(
        kind="expectation_verdict", table="substrate_reverie_thought", ref=row.get("thought_id"),
        at=row.get("expectation_scored_at"), tz_name=tz_name, verdict=str(row["expectation_verdict"]),
        related_refs=[str(chain)] if chain else [],
        correlation_id=row.get("correlation_id") or None,
        payload={"committed_at": _iso(row.get("created_at"))},
    )


HISTORY_ADAPTERS: dict[str, Callable[[Row, str], TemporalSelfEventV1 | None]] = {
    "reverie_chain": reverie_chain,
    "expectation_verdict": expectation_verdict,
}
