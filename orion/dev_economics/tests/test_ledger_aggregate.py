from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.dev_economics.claude_code_ingest import SessionUsageRecord
from orion.dev_economics.ledger_aggregate import (
    aggregate_session_deltas,
    aggregate_session_records,
    diff_session_record,
    has_real_delta,
)

NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


def _record(
    *,
    session_id: str = "s1",
    is_subagent: bool = False,
    models: tuple[str, ...] = ("claude-sonnet-5",),
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    assistant_word_count: int = 100,
    human_word_count: int = 20,
    started_at: datetime | None = NOW,
) -> SessionUsageRecord:
    return SessionUsageRecord(
        session_id=session_id,
        transcript_path=Path(f"/fake/{session_id}.jsonl"),
        is_subagent=is_subagent,
        models=models,
        effort_tiers=("medium",),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        assistant_turn_count=3,
        assistant_word_count=assistant_word_count,
        human_turn_count=2,
        human_word_count=human_word_count,
        started_at=started_at,
        ended_at=started_at,
    )


# ── aggregate_session_records (still used by scripts/replay_dev_economics_ledger.py's
#    whole-history-at-once use case, unaffected by the windowing fix) ──────


def test_empty_window_reports_real_zeros_and_none_cost() -> None:
    agg = aggregate_session_records([])
    assert agg.session_count == 0
    assert agg.total_tokens == 0
    assert agg.total_estimated_cost_usd is None
    assert agg.priced_session_count == 0
    assert agg.unpriced_session_count == 0


def test_priceable_record_sums_real_cost() -> None:
    agg = aggregate_session_records([_record(input_tokens=1_000_000, output_tokens=1_000_000)])
    assert agg.total_estimated_cost_usd is not None
    assert agg.total_estimated_cost_usd > 0
    assert agg.priced_session_count == 1
    assert agg.unpriced_session_count == 0


def test_unpriced_model_excluded_from_cost_but_counted_as_real_usage() -> None:
    agg = aggregate_session_records([_record(models=("xai/grok-4.5",))])
    assert agg.total_tokens == 1500
    assert agg.total_estimated_cost_usd is None
    assert agg.priced_session_count == 0
    assert agg.unpriced_session_count == 1


def test_subagent_records_counted_separately_from_top_level_session_count() -> None:
    agg = aggregate_session_records(
        [
            _record(session_id="parent", is_subagent=False),
            _record(session_id="child", is_subagent=True),
        ]
    )
    assert agg.session_count == 1
    assert agg.subagent_transcript_count == 1
    assert agg.total_tokens == 3000


def test_model_mix_counts_every_record() -> None:
    agg = aggregate_session_records(
        [
            _record(session_id="a", models=("claude-sonnet-5",)),
            _record(session_id="b", models=("claude-sonnet-5",)),
            _record(session_id="c", models=("claude-haiku-4-5-20251001",)),
        ]
    )
    assert agg.model_mix == {"claude-sonnet-5": 2, "claude-haiku-4-5-20251001": 1}


# ── diff_session_record / has_real_delta / aggregate_session_deltas
#    (the live producer's real code path -- regression coverage for the
#    2026-08-12 code review finding) ────────────────────────────────────


def test_diff_against_none_previous_is_the_full_cumulative_total() -> None:
    """A brand-new session (never seen before) diffs as its whole current
    total -- real "new since we started watching" activity."""
    current = _record(input_tokens=1000, output_tokens=500)
    delta = diff_session_record(None, current)
    assert delta.input_tokens == 1000
    assert delta.output_tokens == 500


def test_diff_against_same_totals_is_zero() -> None:
    """A session that was re-scanned but had no new activity since the last
    tick -- the file didn't grow, so the delta must be exactly zero, not a
    repeat of its cumulative total."""
    record = _record(input_tokens=1000, output_tokens=500)
    delta = diff_session_record(record, record)
    assert delta.input_tokens == 0
    assert delta.output_tokens == 0
    assert has_real_delta(delta) is False


def test_diff_captures_real_growth_across_a_multi_tick_session() -> None:
    """The exact bug the 2026-08-12 code review caught: a session that's
    still open across two ticks must have its SECOND tick's growth counted
    too, not silently dropped after the first."""
    tick1 = _record(input_tokens=1000, output_tokens=500, assistant_word_count=100)
    tick2 = _record(input_tokens=2500, output_tokens=1300, assistant_word_count=250)

    delta1 = diff_session_record(None, tick1)
    delta2 = diff_session_record(tick1, tick2)

    assert delta1.input_tokens == 1000
    assert delta2.input_tokens == 1500  # real growth since tick1, not the full 2500
    assert delta2.output_tokens == 800
    assert delta2.assistant_word_count == 150
    assert has_real_delta(delta2) is True


def test_diff_never_goes_negative_on_a_non_monotonic_rewrite() -> None:
    """A transcript file truncated/rewritten between ticks (e.g. Claude Code
    repairing a corrupted transcript) must not produce a negative delta that
    corrupts the running aggregate."""
    previous = _record(input_tokens=5000, output_tokens=2000)
    current = _record(input_tokens=100, output_tokens=50)  # file shrank
    delta = diff_session_record(previous, current)
    assert delta.input_tokens == 0
    assert delta.output_tokens == 0


def test_aggregate_deltas_prices_at_the_tick_observed_time_not_session_start() -> None:
    agg = aggregate_session_deltas(
        [diff_session_record(None, _record(input_tokens=1_000_000, output_tokens=1_000_000))],
        observed_at=NOW,
    )
    assert agg.total_estimated_cost_usd is not None
    assert agg.total_estimated_cost_usd > 0
    assert agg.priced_session_count == 1


def test_aggregate_deltas_unpriced_model_still_counts_real_tokens() -> None:
    agg = aggregate_session_deltas(
        [diff_session_record(None, _record(models=("xai/grok-4.5",)))],
        observed_at=NOW,
    )
    assert agg.total_tokens == 1500
    assert agg.total_estimated_cost_usd is None
    assert agg.unpriced_session_count == 1


def test_aggregate_deltas_empty_list_reports_real_zeros_and_none_cost() -> None:
    agg = aggregate_session_deltas([], observed_at=NOW)
    assert agg.session_count == 0
    assert agg.total_tokens == 0
    assert agg.total_estimated_cost_usd is None
