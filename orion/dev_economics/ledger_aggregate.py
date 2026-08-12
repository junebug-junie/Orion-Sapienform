"""Pure aggregation over ``SessionUsageRecord``s
(orion/dev_economics/claude_code_ingest.py) into one windowed summary --
the same shape ``scripts/replay_dev_economics_ledger.py``'s offline replay
already computes by hand, factored out here so the live producer
(services/orion-cocreation-signals/app/producers/dev_economics.py) and any
future replay/eval script share one real implementation instead of two
copies drifting apart.

No I/O here -- callers pass in an already-filtered iterable of records
(e.g. ``iter_all_session_usage_records`` filtered to a real time window).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from orion.dev_economics.claude_code_ingest import SessionUsageRecord
from orion.dev_economics.pricing import estimate_session_cost_usd


def _record_cost_usd(record: SessionUsageRecord) -> float | None:
    """Same "first-seen model" convention as
    ``scripts/replay_dev_economics_ledger.py``'s own ``_record_cost_usd`` --
    a record touching more than one distinct model (a mid-session model
    switch) is priced against its first-seen model only, a real, disclosed
    (via ``unpriced_session_count`` below) simplification rather than a
    fabricated blended number."""
    if not record.models or record.started_at is None:
        return None
    rate = estimate_session_cost_usd(
        model=record.models[0],
        observed_at=record.started_at,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cache_creation_input_tokens=record.cache_creation_input_tokens,
        cache_read_input_tokens=record.cache_read_input_tokens,
    )
    return rate.total_usd if rate is not None else None


@dataclass(frozen=True)
class DevEconomicsAggregate:
    """One windowed ledger summary. ``total_estimated_cost_usd`` sums only
    the records that could actually be priced (a real Anthropic rate window
    covering that model/date) -- it is a real, honest partial total, not a
    fabricated whole-window number, and ``unpriced_session_count`` discloses
    exactly how many records it excludes so a consumer can judge how much to
    trust it. ``total_estimated_cost_usd`` is ``None`` only when there are
    zero records with *any* real usage in the window (nothing to sum), never
    a fabricated ``0.0`` for "some usage, none priceable"."""

    session_count: int
    subagent_transcript_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_input_tokens: int
    total_cache_read_input_tokens: int
    total_tokens: int
    total_assistant_word_count: int
    total_human_word_count: int
    total_estimated_cost_usd: float | None
    priced_session_count: int
    unpriced_session_count: int
    model_mix: dict[str, int]


def aggregate_session_records(records: Iterable[SessionUsageRecord]) -> DevEconomicsAggregate:
    records = list(records)
    top_level = [r for r in records if not r.is_subagent]
    subagent = [r for r in records if r.is_subagent]

    model_counter: Counter[str] = Counter()
    for r in records:
        model_counter.update(r.models)

    costs = [_record_cost_usd(r) for r in records]
    priced = [c for c in costs if c is not None]

    return DevEconomicsAggregate(
        session_count=len(top_level),
        subagent_transcript_count=len(subagent),
        total_input_tokens=sum(r.input_tokens for r in records),
        total_output_tokens=sum(r.output_tokens for r in records),
        total_cache_creation_input_tokens=sum(r.cache_creation_input_tokens for r in records),
        total_cache_read_input_tokens=sum(r.cache_read_input_tokens for r in records),
        total_tokens=sum(r.total_tokens for r in records),
        total_assistant_word_count=sum(r.assistant_word_count for r in records),
        total_human_word_count=sum(r.human_word_count for r in top_level),
        # None when nothing in the window could be priced -- distinct from a
        # real $0.00 (e.g. "priced but zero token usage", which can't
        # actually happen here since a zero-usage record never yields a
        # SessionUsageRecord at all -- see parse_session_usage_record's own
        # docstring). Deliberately NOT `sum(priced) if records else None`:
        # that would silently report a fabricated $0.00 whenever real
        # records existed but none were priceable (e.g. an all-non-
        # Anthropic-model window), which is exactly the "no empty-shell
        # cognition" failure mode root CLAUDE.md's metric-quality gate
        # warns about.
        total_estimated_cost_usd=(sum(priced) if priced else None),
        priced_session_count=len(priced),
        unpriced_session_count=len(records) - len(priced),
        model_mix=dict(model_counter.most_common()),
    )


@dataclass(frozen=True)
class SessionUsageDelta:
    """Growth in one transcript file's real cumulative totals between two
    checks -- see ``diff_session_record`` for why this exists instead of
    windowing ``SessionUsageRecord`` by ``started_at`` directly (code review
    2026-08-12 caught the window-membership approach silently under-
    reporting every multi-tick session after its first tick)."""

    session_id: str
    is_subagent: bool
    models: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    assistant_word_count: int
    human_word_count: int


def diff_session_record(
    previous: SessionUsageRecord | None, current: SessionUsageRecord
) -> SessionUsageDelta:
    """Real growth in ``current``'s cumulative totals since ``previous`` (the
    same transcript file, last seen at an earlier tick) -- ``previous=None``
    means this file wasn't seen before (a brand-new session, or the very
    first cold-start scan), so its entire cumulative total *would* be
    real "new since we started watching" activity -- but the cold-start
    tick deliberately skips publishing at all (same reasoning as
    ``git_delta_loop``'s own cold start), so in practice a delta against
    ``previous=None`` is only ever computed for a session that started
    mid-flight after the cold-start scan already ran.

    ``max(0, ...)`` guards against a real, if rare, non-monotonic case: a
    transcript file being truncated/rewritten between ticks (e.g. Claude
    Code repairing a corrupted transcript) would otherwise produce a
    negative "delta" that silently corrupts the running aggregate."""

    def _delta(field: str) -> int:
        prev_val = getattr(previous, field) if previous is not None else 0
        return max(0, getattr(current, field) - prev_val)

    return SessionUsageDelta(
        session_id=current.session_id,
        is_subagent=current.is_subagent,
        models=current.models,
        input_tokens=_delta("input_tokens"),
        output_tokens=_delta("output_tokens"),
        cache_creation_input_tokens=_delta("cache_creation_input_tokens"),
        cache_read_input_tokens=_delta("cache_read_input_tokens"),
        assistant_word_count=_delta("assistant_word_count"),
        human_word_count=_delta("human_word_count"),
    )


def has_real_delta(delta: SessionUsageDelta) -> bool:
    """A tick where a session's file was rescanned but nothing new was
    written (no real activity since the last check) must not be counted as
    a real observation -- same "no-op tick, not a spurious zero" reasoning
    as ``git_delta_loop``'s own no-change branch."""
    return bool(
        delta.input_tokens
        or delta.output_tokens
        or delta.cache_creation_input_tokens
        or delta.cache_read_input_tokens
        or delta.assistant_word_count
        or delta.human_word_count
    )


def _delta_cost_usd(delta: SessionUsageDelta, observed_at) -> float | None:
    """Same "first-seen model" convention as ``_record_cost_usd`` above --
    priced against the current rate at the time of *this tick* (real time
    the delta was actually observed), not the session's original start
    time, since a long session's later token growth should be priced at
    whatever rate was current when that growth actually happened."""
    if not delta.models:
        return None
    rate = estimate_session_cost_usd(
        model=delta.models[0],
        observed_at=observed_at,
        input_tokens=delta.input_tokens,
        output_tokens=delta.output_tokens,
        cache_creation_input_tokens=delta.cache_creation_input_tokens,
        cache_read_input_tokens=delta.cache_read_input_tokens,
    )
    return rate.total_usd if rate is not None else None


def aggregate_session_deltas(
    deltas: Iterable[SessionUsageDelta], *, observed_at
) -> DevEconomicsAggregate:
    """Same shape as ``aggregate_session_records`` above, but over real
    incremental growth (``SessionUsageDelta``) rather than whole-file
    cumulative totals -- see ``diff_session_record``'s own docstring for why
    the live producer needs this instead."""
    deltas = list(deltas)
    top_level = [d for d in deltas if not d.is_subagent]
    subagent = [d for d in deltas if d.is_subagent]

    model_counter: Counter[str] = Counter()
    for d in deltas:
        model_counter.update(d.models)

    costs = [_delta_cost_usd(d, observed_at) for d in deltas]
    priced = [c for c in costs if c is not None]

    def _total(field: str) -> int:
        return sum(getattr(d, field) for d in deltas)

    return DevEconomicsAggregate(
        session_count=len(top_level),
        subagent_transcript_count=len(subagent),
        total_input_tokens=_total("input_tokens"),
        total_output_tokens=_total("output_tokens"),
        total_cache_creation_input_tokens=_total("cache_creation_input_tokens"),
        total_cache_read_input_tokens=_total("cache_read_input_tokens"),
        total_tokens=(
            _total("input_tokens")
            + _total("output_tokens")
            + _total("cache_creation_input_tokens")
            + _total("cache_read_input_tokens")
        ),
        total_assistant_word_count=_total("assistant_word_count"),
        total_human_word_count=sum(d.human_word_count for d in top_level),
        total_estimated_cost_usd=(sum(priced) if priced else None),
        priced_session_count=len(priced),
        unpriced_session_count=len(deltas) - len(priced),
        model_mix=dict(model_counter.most_common()),
    )
