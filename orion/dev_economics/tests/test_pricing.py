from __future__ import annotations

from datetime import date, datetime, timezone

from orion.dev_economics.pricing import estimate_session_cost_usd, find_rate


def test_find_rate_sonnet_5_intro_window() -> None:
    rate = find_rate("claude-sonnet-5", date(2026, 8, 11))
    assert rate is not None
    assert rate.input_per_mtok == 2.00
    assert rate.output_per_mtok == 10.00


def test_find_rate_sonnet_5_standard_window_after_intro_ends() -> None:
    rate = find_rate("claude-sonnet-5", date(2026, 9, 15))
    assert rate is not None
    assert rate.input_per_mtok == 3.00
    assert rate.output_per_mtok == 15.00


def test_find_rate_sonnet_5_exact_boundary_dates() -> None:
    """The single most bug-prone point in this module (an off-by-one in
    covers()'s </> comparisons) -- assert the exact transition dates
    directly, not just dates safely on either side."""
    last_intro_day = find_rate("claude-sonnet-5", date(2026, 8, 31))
    assert last_intro_day is not None
    assert last_intro_day.input_per_mtok == 2.00  # still the intro rate

    first_standard_day = find_rate("claude-sonnet-5", date(2026, 9, 1))
    assert first_standard_day is not None
    assert first_standard_day.input_per_mtok == 3.00  # standard rate begins


def test_find_rate_matches_dated_suffix_by_prefix() -> None:
    # Real corpus shape: "claude-haiku-4-5-20251001", not the bare "claude-haiku-4-5"
    rate = find_rate("claude-haiku-4-5-20251001", date(2026, 8, 11))
    assert rate is not None
    assert rate.input_per_mtok == 1.00


def test_find_rate_returns_none_for_unpriced_model() -> None:
    # Real corpus also shows non-Anthropic models this table deliberately
    # does not price.
    assert find_rate("xai/grok-4.5", date(2026, 8, 11)) is None
    assert find_rate("openai/o3", date(2026, 8, 11)) is None


def test_estimate_session_cost_usd_real_numbers() -> None:
    estimate = estimate_session_cost_usd(
        model="claude-sonnet-5",
        observed_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert estimate is not None
    # intro rate: input $2.00/MTok, output $10.00/MTok
    assert estimate.input_cost_usd == 2.00
    assert estimate.output_cost_usd == 10.00
    assert estimate.cache_write_cost_usd == 2.00 * 1.25
    assert estimate.cache_read_cost_usd == 2.00 * 0.1
    assert estimate.total_usd == 2.00 + 10.00 + 2.50 + 0.20


def test_estimate_session_cost_usd_returns_none_for_unpriced_model() -> None:
    assert (
        estimate_session_cost_usd(
            model="moonshotai/kimi-k2.6",
            observed_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
            input_tokens=100,
            output_tokens=100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        is None
    )


def test_estimate_session_cost_usd_returns_none_before_earliest_known_window() -> None:
    # A date before every table entry's effective_from must not silently
    # borrow a later rate -- no window covers it, so no estimate is real.
    assert (
        estimate_session_cost_usd(
            model="claude-haiku-4-5",
            observed_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            input_tokens=100,
            output_tokens=100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        is None
    )


def test_find_rate_picks_latest_effective_from_on_overlap(monkeypatch) -> None:
    """find_rate() must enforce "newest wins" itself, not rely on
    PRICING_TABLE's own literal ordering -- confirmed by code review
    2026-08-11 as a real (if currently dormant) risk: the table happens to
    list claude-sonnet-5's two real windows oldest-first, which only
    produces the right answer today because they're non-overlapping. This
    test constructs a deliberately overlapping pair, in table order that
    would give the WRONG answer if find_rate just returned the first
    match, to lock in that it doesn't."""
    import orion.dev_economics.pricing as pricing_module

    overlapping = (
        pricing_module.PricingRate(
            model_prefix="test-model",
            input_per_mtok=1.00,
            output_per_mtok=1.00,
            effective_from=date(2026, 1, 1),
            effective_until=date(2026, 12, 31),
            source="test: older, listed first",
        ),
        pricing_module.PricingRate(
            model_prefix="test-model",
            input_per_mtok=9.00,
            output_per_mtok=9.00,
            effective_from=date(2026, 6, 1),
            effective_until=None,
            source="test: newer, listed second, overlaps the first",
        ),
    )
    monkeypatch.setattr(pricing_module, "PRICING_TABLE", overlapping)
    rate = pricing_module.find_rate("test-model", date(2026, 7, 1))
    assert rate is not None
    assert rate.input_per_mtok == 9.00  # the newer window, not the first-listed one


def test_estimate_session_cost_usd_open_ended_window_covers_far_future() -> None:
    # Deliberate, honest behavior: an entry with effective_until=None (the
    # newest known rate) is treated as still current for any later date --
    # not a "stale table" bug, a real open-ended window.
    assert (
        estimate_session_cost_usd(
            model="claude-haiku-4-5",
            observed_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
            input_tokens=100,
            output_tokens=100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        is not None
    )
