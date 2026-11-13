"""Hand-written fixtures in the shapes the live system produces (attention patch 3d tests).

Shapes read (read-only) from live Postgres conjourney on 2026-11-11, values synthetic:

* ``want_ledger.decision->'want'``: the latest want per family over 3 days covered 18 families. Every one
  carries the same key set (``LIVE_WANT_KEYS``); event wants (hub.world_pulse_read, vision.ask) add
  ``event_ref`` and ``salience``. Market-filled fields arrive as null.
* ``dream_compaction_delta.delta_json``: 115 staged REM deltas, about 2 a day; every one in the last 14
  days has only ``consolidate`` gist cards (0 downscale, 0 prune entries), 37..114 cards a day.
"""
from __future__ import annotations

from orion.schemas.compaction import ConsolidateEntryV1, MemoryCompactionDeltaV1, PruneEntryV1, DownscaleEntryV1
from orion.schemas.physiology_want import WantV1

LIVE_WANT_KEYS = {"correlation_id", "cost", "family", "graded", "intensity", "lanes", "object_ref", "opened_at",
                  "outcome_value", "producer", "schema_version", "tier", "value_rank", "verb", "want_id", "window"}


def _w(family: str, producer: str, verb: str, tier: int, object_ref, i: int, **kw) -> dict:
    wid = f"00000000-0000-5000-8000-{i:012d}"
    out = {"schema_version": "physiology.want.v1", "want_id": wid, "family": family, "producer": producer,
           "verb": verb, "object_ref": object_ref, "tier": tier, "opened_at": "2026-11-10T07:56:40.041118Z",
           "intensity": None, "outcome_value": None, "value_rank": None, "graded": None, "cost": None,
           "window": {"earliest": None, "latest": "2026-11-10T14:16:44.210025Z"}, "lanes": [],
           "correlation_id": wid}
    out.update(kw)
    return out


# One per live family shape: producers and object_ref forms as the live ledger carries them.
LIVE_WANTS = [
    _w("dream.cycle", "orion-dream.dream_cycle.binding", "make", 1, "dream_cycle:dc-000000000001", 1),
    _w("durable.curiosity_investigate", "orion-durable-runs.admission.binding", "infer", 2,
       "durable_run:000000000002", 2),
    _w("durable.self_inquiry", "orion-durable-runs.admission", "ask", 2, "durable_run:000000000003", 3),
    _w("actions.journal", "orion-actions.journal", "make", 1, "journal_trigger:daily_summary", 4),
    _w("hub.outreach", "orion-hub.outreach.binding", "say", 2, "juniper", 5),
    _w("thought.painting", "orion-thought.painting_self_start", "make", 2, "durable_run:reverie-visual-000000000006", 6),
    _w("hub.world_pulse_read", "orion-hub.reading_events", "look", 2, "https://example.org/paper-7.pdf", 7,
       event_ref="world_pulse_read_seed:reading:00000000-0000-5000-8000-000000000007", salience=0.62),
    _w("vision.ask", "orion-sql-writer.ask_events", "ask", 2, "vision_individual:000000000008", 8,
       event_ref="orion_ask:ask-000000000008", salience=0.41),
    _w("prime", "orion-mind.prime", "infer", 2, None, 9),
]


def live_wants() -> list[WantV1]:
    return [WantV1.model_validate(w) for w in LIVE_WANTS]


def live_delta(delta_id: str = "compaction-delta:00000000-0000-4000-8000-000000000001", cards: int = 3,
               *, downscale: int = 0, prune: int = 0) -> MemoryCompactionDeltaV1:
    """A staged REM delta as live ones look: consolidate gist cards from settled open loops, nothing else
    (``downscale``/``prune`` add entries for the plans live deltas do not reach yet)."""
    return MemoryCompactionDeltaV1(
        delta_id=delta_id, dream_id="dc-000000000001",
        consolidate=[ConsolidateEntryV1(gist_card=f"Consolidate theme 'open-loop-{i:012x}' from 2 settled episode(s).",
                                        evidence_refs=[f"00000000-0000-4000-8000-{i:06d}{j:06d}" for j in range(2)])
                     for i in range(cards)],
        downscale=[DownscaleEntryV1(target_id=f"edge-{i}", old_w=0.9, new_w=0.45, reason="settled") for i in range(downscale)],
        prune=[PruneEntryV1(episodic_id=f"ep-{i}", salience=0.04, ttl_reason="stale") for i in range(prune)],
    )
