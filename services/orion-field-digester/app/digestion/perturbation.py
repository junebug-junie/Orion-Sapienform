from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.field_state import FieldStateV1

from app.ingest.state_deltas import Perturbation

# recent_perturbations / recent_perturbation_count saturating-counter fix
# (2026-07-16). Previously this list was hard-capped to the last 20 distinct
# labels EVER seen (`state.recent_perturbations[-20:]`), with self_state's
# recent_perturbation_count = min(1.0, len(...) / 20.0). Live evidence from a
# 69h/122k-row corpus scan (/mnt/telemetry/field_channels/corpus/
# field_channels.jsonl): the metric read exactly 1.0 (std<1e-6) for the ENTIRE
# window, starting from the very first captured tick. Root cause confirmed
# against orion-field-digester's own substrate_field_applied_deltas table
# (Postgres, 2026-07-16 20:28-21:10 UTC, 5063 rows): quiet baseline is ~6-10
# distinct applied deltas/minute, but real traffic is bursty -- observed
# minute-buckets of 505, 557, 582, 729, 900 distinct deltas/minute (batch
# backlog drains). At the OLD cap of 20, even the quiet baseline saturates the
# list within ~2-3 minutes, and any burst blows through 20 in a couple of
# seconds -- and because nothing ever expired an entry, once saturated the
# count stayed pinned at 1.0 forever, regardless of whether real activity
# later dropped back to near-zero.
#
# Fix: prune by wall-clock recency instead of a fixed label count, using the
# new parallel `recent_perturbation_at` field (orion/schemas/field_state.py).
# A window of 60s was chosen to actually distinguish the two observed
# regimes: quiet baseline (~6-10/min => ~6-10 entries in a 60s window, well
# under saturation) vs. a genuine burst (hundreds/min => saturates quickly,
# same as before) -- but a burst's elevated reading now decays back down
# within 60s of the burst ending, which the old mechanism could never do.
RECENT_PERTURBATION_WINDOW_SECONDS = 60.0
# Defensive backstop only, not a normal operating cap. At the highest
# observed live rate (~900/min = 15/s) a full 60s window could in principle
# hold ~900 entries; this just bounds worst-case unbounded growth from a
# runaway producer bug so the list can never grow without limit.
RECENT_PERTURBATION_MAX_ENTRIES = 2000


def apply_perturbations(
    state: FieldStateV1,
    perturbations: list[Perturbation],
    *,
    now: datetime | None = None,
) -> FieldStateV1:
    # Default to state.generated_at, NOT wall-clock datetime.now(). worker.py's
    # _tick() already sets state.generated_at = now (the real tick timestamp)
    # immediately before calling run_digestion_tick() -> apply_perturbations(),
    # so this stays semantically correct in production while remaining fully
    # deterministic for replay: tests/test_field_deterministic_replay.py feeds
    # the same fixed `now` into empty_field_state() and expects byte-identical
    # output for identical input receipts across repeated runs of
    # run_digestion_tick() -- a wall-clock default broke that (each run got a
    # different real-time stamp, chosen with the `or` operator this replaces).
    ts = now if now is not None else state.generated_at
    for p in perturbations:
        node_vec = state.node_vectors.setdefault(p.node_id, {})
        if p.mode == "replace":
            node_vec[p.channel] = max(0.0, min(1.0, p.intensity))
        elif p.channel == "availability":
            node_vec[p.channel] = min(node_vec.get(p.channel, 1.0), p.intensity)
        else:
            node_vec[p.channel] = min(1.0, node_vec.get(p.channel, 0.0) + p.intensity)
        if p.label not in state.recent_perturbations:
            state.recent_perturbations.append(p.label)
            state.recent_perturbation_at.append(ts)
    _prune_recent_perturbations(state, now=ts)
    return state


def _prune_recent_perturbations(state: FieldStateV1, *, now: datetime) -> None:
    """Keep only labels seen within RECENT_PERTURBATION_WINDOW_SECONDS of
    `now`, replacing the old "last 20 distinct labels ever" cap.
    `recent_perturbations` and `recent_perturbation_at` are always the same
    length and index-aligned -- both are appended together above, so they are
    pruned together here too.
    """
    cutoff = now - timedelta(seconds=RECENT_PERTURBATION_WINDOW_SECONDS)
    kept_labels: list[str] = []
    kept_at: list[datetime] = []
    for label, at in zip(state.recent_perturbations, state.recent_perturbation_at):
        at_aware = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
        if at_aware >= cutoff:
            kept_labels.append(label)
            kept_at.append(at)
    if len(kept_labels) > RECENT_PERTURBATION_MAX_ENTRIES:
        kept_labels = kept_labels[-RECENT_PERTURBATION_MAX_ENTRIES:]
        kept_at = kept_at[-RECENT_PERTURBATION_MAX_ENTRIES:]
    state.recent_perturbations = kept_labels
    state.recent_perturbation_at = kept_at
