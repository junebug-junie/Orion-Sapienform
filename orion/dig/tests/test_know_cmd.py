"""``python -P -m orion.dig know <words>`` (issue #3535): a thin client of orion-recall's
``POST /lived/know``. No network: the HTTP post is a fake."""
from __future__ import annotations

import json
import urllib.error

import pytest

from orion.dig import __main__ as dig
from orion.dig import know
from orion.dig.brief import DIG_BRIEF_LINE

HIT = {"strength": "strong", "kind": "episode_memory", "purpose": "about_juniper", "id": "episode_memory:abc",
       "date": "2026-11-10", "text": "Juniper said the dog came home with stitches.", "fit": 0.91}
WEAK = {"strength": "weak", "kind": "person_fact", "purpose": "person_fact", "id": "person_fact:7",
        "date": "undated", "text": "Biscuit, a dog, " + "x" * 400, "fit": 0.002}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    return tmp_path


def _poster(answer, seen=None):
    def post(url, payload):
        if seen is not None:
            seen.append((url, payload))
        if isinstance(answer, Exception):
            raise answer
        return answer
    return post


def test_lookup_posts_the_words_and_the_turn_id():
    seen: list = []
    env = {"ORION_TURN_CORRELATION_ID": "turn-1", "RECALL_SERVICE_URL": "http://127.0.0.1:8260/"}
    know.lookup("  pet   stitches ", post=_poster({"status": "ok", "hits": []}, seen), environ=env)
    assert seen == [("http://127.0.0.1:8260/lived/know", {"words": "pet stitches", "turn_id": "turn-1"})]


def test_recall_url_order():
    assert know.recall_url({}) == "http://orion-athena-recall:8260"
    assert know.recall_url({"RECALL_SERVICE_URL": "http://a:1"}) == "http://a:1"
    assert know.recall_url({"RECALL_SERVICE_URL": "http://a:1", "ORION_DIG_RECALL_URL": "http://b:2"}) == "http://b:2"


def test_no_turn_id_outside_a_turn():
    seen: list = []
    know.lookup("pet stitches", post=_poster({"status": "ok"}, seen), environ={})
    assert "turn_id" not in seen[0][1]


@pytest.mark.parametrize("exc", [urllib.error.URLError("refused"), TimeoutError(),
                                 urllib.error.HTTPError("u", 503, "off", {}, None)])
def test_unreachable_recall_is_never_no_memory(exc):
    with pytest.raises(know.KnowError, match="NOT the same as having no memory"):
        know.lookup("pet stitches", post=_poster(exc), environ={})


def test_render_strong_and_weak_are_labelled():
    lines = know.render("pet stitches", {"status": "ok", "backend": "hybrid", "hits": [HIT, WEAK]})
    text = "\n".join(lines)
    assert lines[0] == 'know "pet stitches": 1 strong, 1 weak'
    assert "(episode_memory/about_juniper · episode_memory:abc · 2026-11-10) Juniper said" in text
    assert "Weak (" in text and "below the bar" in text
    assert "(person_fact · person_fact:7 · undated)" in text
    assert max(len(ln) for ln in lines) < 320  # long statements are cut on screen (full text in the file)
    assert "No stored memory" not in text


def test_render_never_says_nothing_found_when_weak_hits_exist():
    text = "\n".join(know.render("pet", {"status": "below_bar", "backend": "hybrid", "hits": [WEAK]}))
    assert "No stored memory" not in text and "Weak (" in text


def test_render_empty_hybrid_says_try_other_words():
    text = "\n".join(know.render("zzz", {"status": "below_bar", "backend": "hybrid", "hits": []}))
    assert "No stored memory" in text and "Try other words" in text


@pytest.mark.parametrize("status", ["timeout", "error", "off", "busy"])
def test_render_failure_is_not_no_memory(status):
    text = "\n".join(know.render("pet", {"status": status, "backend": "", "hits": []}))
    assert "NOT the same as having no memory" in text and "No stored memory" not in text


def test_render_degraded_is_labelled():
    text = "\n".join(know.render("pet", {"status": "degraded", "backend": "degraded",
                                         "degraded": ["rerank_timeout"], "hits": [HIT]}))
    assert "DEGRADED: searched by exact words only (rerank_timeout)" in text


def test_cmd_know_writes_the_full_record_and_prints_a_short_one(home, monkeypatch):
    monkeypatch.setattr(know, "_post_json", lambda url, payload: {"status": "ok", "backend": "hybrid",
                                                                   "hits": [HIT, WEAK], "corr_id": "c"})
    out = dig.cmd_know(["pet", "stitches"], dig.Out("d1"))
    assert "1 strong, 1 weak" in out and "full record" in out
    record = json.loads((home / "dig" / "d1" / "know-1.json").read_text())
    assert record["words"] == "pet stitches" and record["hits"][1]["text"].endswith("x" * 50)


def test_main_dispatches_know_and_reports_failure(home, monkeypatch, capsys):
    monkeypatch.setattr(know, "_post_json", lambda url, payload: {"status": "ok", "backend": "hybrid", "hits": [HIT]})
    assert dig.main(["know", "pet", "stitches"]) == 0
    assert "episode_memory:abc" in capsys.readouterr().out

    def down(url, payload):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(know, "_post_json", down)
    assert dig.main(["know", "pet"]) == 1
    assert "NOT the same as having no memory" in capsys.readouterr().err
    assert dig.main(["know"]) == 2  # no words: usage


def test_help_lists_know():
    assert "know <words>" in dig.HELP


def test_brief_carries_look_before_you_deny():
    """Recall rev 2: Orion does not say "I have no memory of that" until a search has run."""
    assert "Before telling Juniper you don't remember or have no record" in DIG_BRIEF_LINE
    assert "`orion.dig know <words>`" in DIG_BRIEF_LINE


def test_render_degraded_empty_never_says_no_stored_memory():
    text = "\n".join(know.render("pet", {"status": "degraded", "backend": "degraded",
                                         "degraded": ["rerank_unavailable"], "hits": []}))
    assert "DEGRADED" in text and "No stored memory" not in text


def test_render_degraded_hits_are_not_called_above_the_bar():
    text = "\n".join(know.render("pet", {"status": "degraded", "backend": "degraded", "hits": [HIT]}))
    assert "no bar was applied" in text and "cleared recall's bar" not in text


def test_render_unverified_matches_are_said():
    text = "\n".join(know.render("pet", {"status": "unverified", "backend": "hybrid", "hits": [], "unverified": 3}))
    assert "3 more match(es) found but could not be checked" in text and "No stored memory" not in text


def test_brief_says_a_search_that_could_not_run_is_not_no_memory():
    """#3663: a skipped, degraded, timed-out or failed search is not grounds for "I have no memory"."""
    from orion.dig.brief import MAX_BRIEF_CHARS

    sentence = ("If it could not search properly (skipped, degraded, timed out, error), say you could not look, "
                "never that you have no memory.")
    assert sentence in DIG_BRIEF_LINE and DIG_BRIEF_LINE.endswith(sentence)
    assert len(DIG_BRIEF_LINE) <= MAX_BRIEF_CHARS


@pytest.mark.parametrize("status", ["skipped_short", "timeout", "error", "busy", "off", "degraded"])
def test_render_every_non_search_says_not_no_memory(status):
    data = {"status": status, "hits": []}
    if status == "degraded":
        data.update(backend="degraded", degraded=["rerank_timeout"])
    text = "\n".join(know.render("the", data))
    assert "No stored memory" not in text
    assert "NOT the same as having no memory" in text or "does not mean no memory" in text


def test_render_skipped_short_carries_the_not_no_memory_sentence():
    text = "\n".join(know.render("the a", {"status": "skipped_short", "hits": []}))
    assert "nothing was searched" in text and "This is NOT the same as having no memory of it" in text
    assert "use names, places, things" in text
