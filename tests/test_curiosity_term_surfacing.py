"""The deterministic half of Orion's curiosity loop.

The tests that matter here are the REFUSALS. A detector that always finds
something to be curious about manufactures significance every day, and the
resulting journal entries are cognition-shaped output with nothing behind them.
Two of the refusals below are regressions against what the first live run over
the real corpus actually produced.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    """The statistic is a RATE, not a count. The same conversation at ten times
    the volume must not make every word in it look like a new obsession.

    Built WITHOUT the shared `_report` padding on purpose: that helper pads the
    two windows unequally (40 vs 400 filler messages), which by itself dilutes
    the baseline and lifts every real term's recent share. That asymmetry is
    fine for the other fixtures -- it is what makes a genuinely-new term stand
    out -- but it would make this test pass or fail for the wrong reason. Here
    both windows must have IDENTICAL composition and differ only in size, which
    is the actual claim being tested."""
    line = "the usual thing we always talk about here"
    messages = [(YESTERDAY, line) for _ in range(100)]
    messages += [(LONG_AGO, line) for _ in range(1000)]
    report = build_surfacing_report(messages, now=NOW)
    assert not report.underpowered, "fixture must be big enough to be measurable"
    assert report.terms == [], f"same conversation, more of it, surfaced: {_terms(report)}"


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
