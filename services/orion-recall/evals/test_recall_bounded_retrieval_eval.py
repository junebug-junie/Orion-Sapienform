"""Gate form of run_recall_bounded_retrieval_eval.py (see its docstring for
what this can and cannot measure). Stubbed backends only; no network."""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_recall_bounded_retrieval_eval", Path(__file__).resolve().parent / "run_recall_bounded_retrieval_eval.py"
)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


@functools.lru_cache(maxsize=1)
def _report():
    return ev.run_eval()


def test_every_real_query_has_bounded_sub_queries_and_search_text() -> None:
    r = _report()
    assert r["rows"] >= 50
    assert r["max_sub_queries"] <= ev.K + 2
    assert r["max_query_chars_searched"] <= ev.MAX_QUERY_CHARS
    assert r["stopword_sub_queries"] == []


def test_intake_choice_is_bounded_by_length() -> None:
    for row in _report()["results"]:
        expected = "condensed" if row["chars"] > ev.MAX_QUERY_CHARS else "fragment"
        assert row["new"]["source"] == expected, (row["verb"], row["chars"])


def test_context_feeds_run_at_most_once_per_recall() -> None:
    assert _report()["feeds_called_once"] is True


def test_long_queries_finish_fast_and_do_bounded_work() -> None:
    r = _report()
    assert r["long_query_count"] >= 1
    assert r["long_max_wall_ms"] < 5000
    # Work no longer scales with prompt length (old projected: 268 x 7 = 1,876).
    assert r["long_max_calls_new"] <= 6 + (ev.K + 2) + 1
    assert r["long_max_calls_old_projected"] >= 1000


def test_short_query_top8_is_stable_under_the_cap() -> None:
    r = _report()
    assert r["short_query_count"] >= 10
    assert r["short_top8_overlap_mean"] >= 0.9
