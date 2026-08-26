"""The deterministic half of Orion's curiosity loop.

The tests that matter here are the REFUSALS. A detector that always finds
something to be curious about manufactures significance every day, and the
resulting journal entries are cognition-shaped output with nothing behind them.
Two of the refusals below are regressions against what the first live run over
the real corpus actually produced.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.curiosity.term_surfacing import (
    MIN_LIFT,
    MIN_RECENT_COUNT,
    MIN_RECENT_MESSAGES,
    build_surfacing_report,
)

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
YESTERDAY = NOW - timedelta(hours=6)
LONG_AGO = NOW - timedelta(days=8)

# Enough filler that the report is never `underpowered`, and so that a term's
# SHARE of tokens is what moves rather than the raw corpus size.
_FILLER = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sigma"


def _messages(recent: list[str], baseline: list[str]) -> list[tuple[datetime, str]]:
    out = [(YESTERDAY, text) for text in recent]
    out += [(LONG_AGO, text) for text in baseline]
    return out


def _padded(n: int, *, at: datetime) -> list[tuple[datetime, str]]:
    return [(at, _FILLER) for _ in range(n)]


def _report(recent: list[str], baseline: list[str], **kw):
    messages = _messages(recent, baseline)
    messages += _padded(40, at=YESTERDAY) + _padded(400, at=LONG_AGO)
    return build_surfacing_report(messages, now=NOW, **kw)


def _terms(report) -> set[str]:
    return {t.term for t in report.terms}


# --- it finds a real one ---------------------------------------------------


def test_a_term_never_said_before_surfaces() -> None:
    report = _report(
        recent=["foveal probing again", "the foveal path", "foveal window sizing"] * 2,
        baseline=["nothing to do with that here"] * 20,
    )
    assert "foveal" in _terms(report)
    surfaced = next(t for t in report.terms if t.term == "foveal")
    assert surfaced.is_new
    assert surfaced.baseline_count == 0
    assert "not once in the whole baseline window" in surfaced.describe()


def test_a_term_said_far_more_than_usual_surfaces() -> None:
    report = _report(
        recent=["probe probe probe", "probe again", "another probe here"] * 2,
        baseline=["probe once"] + ["unrelated filler text"] * 30,
    )
    assert "probe" in _terms(report)


def test_the_description_always_carries_its_own_bar() -> None:
    """The journal entry and the prompt both quote this. A number without the
    threshold it cleared is not inspectable."""
    report = _report(
        recent=["probe probe probe", "probe again", "more probe"] * 2,
        baseline=["probe once"] + ["filler"] * 30,
    )
    surfaced = next(t for t in report.terms if t.term == "probe")
    assert "expected" in surfaced.describe()
    assert "separate messages" in surfaced.describe()


# --- the refusals ----------------------------------------------------------


def test_an_ordinary_day_surfaces_nothing() -> None:
    """The common case, and it must stay common."""
    report = _report(
        recent=["talking about the usual things"] * 10,
        baseline=["talking about the usual things"] * 100,
    )
    assert report.terms == []
    assert not report.has_signal


def test_a_term_repeated_inside_one_message_does_not_surface() -> None:
    """REGRESSION, first live run 2026-08-25. Worktree directory names pasted
    dozens of times inside a single message dominated the output. A path echoed
    forty times in one message is one topic mentioned once -- requiring the term
    to recur across SEPARATE messages is what separated a subject Juniper kept
    returning to from a string that appeared in a paste."""
    report = _report(
        recent=["widget " * 40],
        baseline=["nothing relevant"] * 30,
    )
    assert "widget" not in _terms(report)


def test_worktree_and_branch_shaped_tokens_are_never_subjects() -> None:
    """REGRESSION, same live run. These are artifacts of how Juniper works, not
    things being discussed."""
    noisy = [
        "orion-sapienform-foveal-probe-via-gateway",
        "services-vision-host-thing",
        "some-hyphenated-branchname",
    ]
    report = _report(
        recent=[" ".join(noisy)] * 6,
        baseline=["clean baseline text"] * 30,
    )
    assert _terms(report) & set(noisy) == set()


def test_the_count_bars_are_pinned_to_their_actual_values() -> None:
    """Review finding 2026-08-26: the two bar tests below express their
    fixtures as `MIN_X - 1`, so they move WITH the constant -- setting
    MIN_RECENT_COUNT to 1 makes the fixture contain zero mentions and the test
    passes vacuously. A test that imports the constant it exists to pin cannot
    pin it."""
    assert MIN_RECENT_COUNT == 5
    assert MIN_RECENT_MESSAGES == 3
    assert MIN_LIFT == 3.0


def test_the_count_bar_is_inclusive_at_exactly_the_threshold() -> None:
    """Off-by-one: `count < MIN_RECENT_COUNT` must reject 4 and accept 5."""
    base = ["unrelated filler words here"] * 60
    at_bar = _report(recent=[f"kestrel circling overhead {i}" for i in range(5)], baseline=base)
    below = _report(recent=[f"kestrel circling overhead {i}" for i in range(4)], baseline=base)
    assert "kestrel" in _terms(at_bar)
    assert "kestrel" not in _terms(below)


def test_the_message_bar_is_inclusive_at_exactly_the_threshold() -> None:
    base = ["unrelated filler words here"] * 60
    at_bar = _report(recent=["kestrel kestrel"] * 3, baseline=base)
    below = _report(recent=["kestrel kestrel kestrel"] * 2, baseline=base)
    assert "kestrel" in _terms(at_bar)
    assert "kestrel" not in _terms(below)


def test_the_lift_arithmetic_is_hand_checkable() -> None:
    """`expected = (baseline_count / baseline_tokens) * recent_tokens`.

    Hand-computed against the numbers the report itself reports, so this fails
    if the formula changes even when the pass/fail verdict happens not to."""
    messages = [(YESTERDAY, "kestrel " * 20)] * 5
    messages += [(YESTERDAY, _FILLER)] * 40
    messages += [(LONG_AGO, "kestrel")] * 10
    messages += [(LONG_AGO, _FILLER)] * 400
    report = build_surfacing_report(messages, now=NOW)
    found = next(t for t in report.terms if t.term == "kestrel")
    expected = (found.baseline_count / report.baseline_tokens) * report.recent_tokens
    assert found.expected_count == pytest.approx(expected)
    assert found.lift == pytest.approx(found.recent_count / expected)
    # And the rate normalisation is genuinely doing work: dropping it (i.e.
    # `expected = baseline_count`) would give a materially different number.
    assert found.expected_count != pytest.approx(float(found.baseline_count))


def test_the_underpowered_thresholds_are_pinned() -> None:
    """Only "always False" was caught before; the specific numbers were not."""
    from orion.curiosity.term_surfacing import build_surfacing_report as build

    just_under = build(
        [(YESTERDAY, _FILLER)] * 16 + [(LONG_AGO, _FILLER)] * 400, now=NOW
    )
    assert just_under.recent_tokens < 200 and just_under.underpowered
    just_over = build(
        [(YESTERDAY, _FILLER)] * 20 + [(LONG_AGO, _FILLER)] * 400, now=NOW
    )
    assert just_over.recent_tokens >= 200 and not just_over.underpowered
    thin_baseline = build(
        [(YESTERDAY, _FILLER)] * 40 + [(LONG_AGO, _FILLER)] * 100, now=NOW
    )
    assert thin_baseline.baseline_tokens < 2000 and thin_baseline.underpowered


def test_window_boundaries_do_not_double_count() -> None:
    """`since <= stamp < until` -- a message exactly on the recent boundary
    belongs to exactly one window."""
    boundary = NOW - timedelta(hours=24)
    messages = [(boundary, "kestrel " * 5)] * 5
    messages += [(YESTERDAY, _FILLER)] * 40 + [(LONG_AGO, _FILLER)] * 400
    report = build_surfacing_report(messages, now=NOW)
    total = report.recent_messages + report.baseline_messages
    assert total == 445, f"a boundary message was counted twice or not at all: {total}"


def test_the_limit_actually_truncates() -> None:
    """The old limit test asserted `<= 4` on a fixture producing fewer than 4,
    so removing the slice entirely survived."""
    words = [
        "kestrel", "gannet", "petrel", "fulmar", "shrike", "merlin",
        "osprey", "curlew", "godwit", "avocet",
    ]
    recent = [f"{w} " * 6 for w in words] * 4
    unlimited = _report(recent=recent, baseline=["filler words here"] * 60, limit=50)
    assert len(unlimited.terms) > 4, "fixture must produce more candidates than the limit"
    limited = _report(recent=recent, baseline=["filler words here"] * 60, limit=4)
    assert len(limited.terms) == 4


def test_a_term_said_only_a_few_times_does_not_surface() -> None:
    said = MIN_RECENT_COUNT - 1
    report = _report(
        recent=[f"kestrel mention {i}" for i in range(said)],
        baseline=["unrelated"] * 30,
    )
    assert "kestrel" not in _terms(report)


def test_a_term_below_the_message_bar_does_not_surface() -> None:
    messages = MIN_RECENT_MESSAGES - 1
    report = _report(
        recent=["kestrel kestrel kestrel kestrel" for _ in range(messages)],
        baseline=["unrelated"] * 30,
    )
    assert "kestrel" not in _terms(report)


def test_a_busy_day_alone_does_not_surface_everything() -> None:
    """The statistic is a RATE, not a count -- `expected = (baseline_count /
    baseline_tokens) * recent_tokens`, the single load-bearing line in the
    module.

    THE RECENT WINDOW MUST BE THE BIG ONE. Review finding 2026-08-26: this test
    was built with the BASELINE as the large window (100 recent vs 1000
    baseline), which means deleting rate normalisation entirely
    (`expected = baseline_count`) still yielded lift = 100/1000 = 0.1, below the
    bar, and the test passed. A "busy day" is by definition a day when the
    RECENT window is the large one -- inverted, the mutant yields lift = 10.0
    and surfaces three words from a conversation that never changed."""
    line = "the usual thing we always talk about here"
    messages = [(YESTERDAY, line) for _ in range(1000)]
    messages += [(LONG_AGO, line) for _ in range(700)]
    report = build_surfacing_report(messages, now=NOW)
    assert not report.underpowered, "fixture must be big enough to be measurable"
    assert report.recent_tokens > report.baseline_tokens, (
        "the RECENT window must be the larger one or this cannot catch the "
        "mutation it exists for"
    )
    assert report.terms == [], f"same conversation, more of it, surfaced: {_terms(report)}"


def test_the_same_conversation_at_lower_volume_also_surfaces_nothing() -> None:
    """The mirror of the above -- a quiet day is not a collapse."""
    line = "the usual thing we always talk about here"
    messages = [(YESTERDAY, line) for _ in range(700)]
    messages += [(LONG_AGO, line) for _ in range(1000)]
    report = build_surfacing_report(messages, now=NOW)
    assert not report.underpowered
    assert report.terms == []


def test_stopwords_never_surface() -> None:
    report = _report(
        recent=["really going to think about this thing"] * 8,
        baseline=["unrelated"] * 30,
    )
    assert _terms(report) & {"really", "going", "think", "thing", "about"} == set()


# --- "not enough to look at" is not "nothing to report" --------------------


def test_a_thin_corpus_reports_underpowered_rather_than_quiet() -> None:
    """A quiet day and a broken transcript reader look identical unless these
    are separate states."""
    report = build_surfacing_report(
        [(YESTERDAY, "two words"), (LONG_AGO, "three more words")], now=NOW
    )
    assert report.underpowered
    assert not report.has_signal


def test_a_real_corpus_is_not_underpowered() -> None:
    report = _report(recent=["ordinary"] * 10, baseline=["ordinary"] * 100)
    assert not report.underpowered


def test_an_empty_corpus_does_not_crash() -> None:
    report = build_surfacing_report([], now=NOW)
    assert report.terms == []
    assert report.underpowered
    assert report.recent_tokens == 0


# --- ranking ---------------------------------------------------------------


def test_ranking_prefers_what_was_actually_said_most_not_highest_lift() -> None:
    """Lift is unbounded for a brand-new term, so ranking on it would put every
    one-off neologism above a subject Juniper genuinely returned to all day."""
    recent = ["kestrel " * 3] * 3 + ["probe " * 10] * 6
    baseline = ["probe"] * 4 + ["filler"] * 40
    report = _report(recent=recent, baseline=baseline)
    assert report.terms, "fixture must surface something"
    assert report.terms[0].term == "probe"
    assert report.terms[0].recent_count > report.terms[-1].recent_count or len(report.terms) == 1


def test_the_limit_is_honoured() -> None:
    recent = [f"term{i} " * 6 for i in range(20)] * 3
    report = _report(recent=recent, baseline=["filler"] * 60, limit=4)
    assert len(report.terms) <= 4


def test_windows_are_reported_so_a_reader_can_check_them() -> None:
    report = _report(recent=["ordinary"] * 10, baseline=["ordinary"] * 100)
    assert report.recent_since < report.generated_at
    assert report.baseline_since < report.recent_since
    assert report.generated_at == NOW
