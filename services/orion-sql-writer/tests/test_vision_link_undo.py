"""Undoing a "same person" link made on a walkway story card (who-card redesign Patch D, 2026-11-24).

Real Postgres (``ORION_TEST_PG_URI``, throwaway: it TRUNCATEs the walkway tables), with the walkway
migrations applied (see tests/test_vision_split.py for the setup).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_individuals as vi  # noqa: E402
from app import vision_link_undo as lu  # noqa: E402

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")
NOW = datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc)
STORY = "f" * 32


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_individual_sighting, "
                          "vision_individual, vision_ask_picture_verdict, vision_individual_split CASCADE"))
    yield engine
    engine.dispose()


def _ind(conn, iid, label=None, *, copied=False, labeled_at=None, ask="ask-typed"):
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, label, labeled_at, label_ask_id, label_copied) "
        "VALUES (:i, 'walkway', 'person', ARRAY[1.0]::real[], 3, :t, :t, 1, 1, :l, :la, :a, :c)"),
        {"i": iid, "t": NOW - timedelta(days=3), "l": label, "la": labeled_at, "a": ask if label else None,
         "c": copied})


def _pair(conn, a, b, verdict="same", ask="ask-typed", at=None):
    from sqlalchemy import text

    lo, hi = sorted((a, b))
    conn.execute(text("INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, "
                      "decided_at) VALUES (:a, :b, :v, :k, :t)"),
                 {"a": lo, "b": hi, "v": verdict, "k": ask, "t": at or NOW - timedelta(hours=1)})


def _card_link(conn, ask_id, a, b, *, at=None, via="motion_story", answer="same"):
    """Her "Same person? yes" on the story card, already applied (the pair row is the ask's)."""
    from sqlalchemy import text

    at = at or NOW - timedelta(hours=1)
    conn.execute(text(
        "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, answered_at, applied_at, "
        "answer_detail) VALUES (:k, 'q', 'vision_same_individual', :r, 'answered', :ans, :t, :t, CAST(:d AS jsonb))"),
        {"k": ask_id, "r": vi.pair_ref(a, b), "ans": answer, "t": at,
         "d": json.dumps({"story_id": STORY, "action": "same_name", "via": via})})
    _pair(conn, a, b, ask=ask_id, at=at)


def _label(conn, iid):
    from sqlalchemy import text

    return tuple(conn.execute(text("SELECT label, label_copied FROM vision_individual WHERE individual_id=:i"),
                              {"i": iid}).one())


@needs_pg
def test_undo_removes_the_link_and_rederives_the_copied_name(pg, caplog) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _ind(conn, "a", "Sasha", labeled_at=NOW - timedelta(days=2))
        _ind(conn, "b", "Sasha", copied=True, labeled_at=NOW - timedelta(days=2))
        _card_link(conn, "ask-link", "a", "b")
    caplog.set_level(logging.INFO, logger="sql-writer.vision_link_undo")
    with pg.begin() as conn:
        rep = lu.undo_link(conn, "ask-link", now=NOW)
    assert rep["ok"] is True and rep["relabelled"] == {"b": None}
    with pg.begin() as conn:
        assert conn.execute(text("SELECT count(*) FROM vision_individual_pair_verdict")).scalar() == 0
        assert _label(conn, "a") == ("Sasha", False) and _label(conn, "b") == (None, False)
        d = conn.execute(text("SELECT answer_detail FROM orion_ask WHERE ask_id='ask-link'")).scalar()
        assert d["undone_at"] == NOW.isoformat()
        assert lu.undo_link(conn, "ask-link", now=NOW)["reason"].startswith("already undone")
    assert any("[VISION_LINK_UNDONE] ask_id=ask-link" in r.message for r in caplog.records)


@needs_pg
def test_undo_refuses_out_of_scope_too_old_and_dependents_and_names_them(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        for i in ("a", "b", "c", "d", "e", "f"):
            _ind(conn, i)
        _card_link(conn, "ask-other-card", "a", "b", via="ask_card")            # not this card's
        _card_link(conn, "ask-old", "c", "d", at=NOW - timedelta(days=8))        # out of the window
        _card_link(conn, "ask-named", "e", "f")
        # A name typed on the story card after the link (a vision_story_person "someone else, called Sam"
        # applied as a typed label): read from the label itself, whichever card wrote it.
        conn.execute(text("UPDATE vision_individual SET label = 'Sam', label_copied = false, labeled_at = :t, "
                          "label_ask_id = 'ask-name-later' WHERE individual_id = 'f'"), {"t": NOW - timedelta(minutes=5)})
    with pg.begin() as conn:
        assert "not a 'same person' link made on a walkway story card" in lu.undo_link(conn, "ask-other-card", now=NOW)["reason"]
        assert lu.undo_link(conn, "ask-old", now=NOW)["reason"].startswith("older than 7 days")
        rep = lu.undo_link(conn, "ask-named", now=NOW)
        assert rep["ok"] is False and rep["name_asks"] == ["ask-name-later"]
        assert conn.execute(text("SELECT count(*) FROM vision_individual_pair_verdict")).scalar() == 3
    with pg.begin() as conn:
        conn.execute(text("UPDATE vision_individual SET label = NULL, labeled_at = NULL, label_ask_id = NULL "
                          "WHERE individual_id = 'f'"))
        conn.execute(text(
            "INSERT INTO vision_individual_split (split_id, source_individual_id, new_individual_id, ask_id, group_key, "
            "status, moved, linked_pair) VALUES ('split-1', 'x', 'e', 'ask-x', '', 'split', '[]'::jsonb, "
            "CAST(:lp AS jsonb))"),
            {"lp": json.dumps([{"individual_a": "e", "individual_b": "f", "verdict": "same", "ask_id": "ask-named"}])})
    with pg.begin() as conn:
        rep = lu.undo_link(conn, "ask-named", now=NOW)
        assert rep["ok"] is False and rep["splits"] == ["split-1"]


@needs_pg
def test_undo_refuses_plainly_when_other_same_links_still_join_the_two(pg) -> None:
    """Orion's question 3: a split's or a typed name's 'same' row can join the same two individuals the
    card's link joins; removing only the card's row would look like an undo that did nothing. Refused,
    naming the rows still joining them. A 'different' inside the group is reported as a contradiction."""
    with pg.begin() as conn:
        for i in ("a", "b", "x"):
            _ind(conn, i)
        _card_link(conn, "ask-link", "a", "b")
        _pair(conn, "a", "x", ask="split-made")
        _pair(conn, "x", "b", ask="name-made")
        _ind(conn, "y")
        _pair(conn, "b", "y", ask="ask-y")
        _pair(conn, "a", "y", verdict="different", ask="ask-diff")
    with pg.begin() as conn:
        rep = lu.undo_link(conn, "ask-link", now=NOW)
    assert rep["ok"] is False and rep["reason"].startswith("still joined through other 'same' links")
    assert rep["still_joined_by"] == ["a|x(ask split-made)", "b|x(ask name-made)"]
    assert rep["contradictions"] == ["a|y:different(ask ask-diff)"]


@needs_pg
def test_deleting_the_middle_link_of_a_chain_gives_each_end_its_own_typed_name(pg) -> None:
    """Orion's question 8: groups a-b-c-d, typed on both ends (a "Ann", d "Annie", typed later), b and c
    carry the newest copy ("Annie"). Undoing the middle link (b-c): b follows a, c follows d; the two
    typed names are never touched. Also the three-group chain: undoing a-b of a-b-c."""
    with pg.begin() as conn:
        _ind(conn, "a", "Ann", labeled_at=NOW - timedelta(days=2), ask="ask-a")
        _ind(conn, "b", "Annie", copied=True, labeled_at=NOW - timedelta(days=1), ask="ask-d")
        _ind(conn, "c", "Annie", copied=True, labeled_at=NOW - timedelta(days=1), ask="ask-d")
        _ind(conn, "d", "Annie", labeled_at=NOW - timedelta(days=1), ask="ask-d")
        _pair(conn, "a", "b")
        _card_link(conn, "ask-middle", "b", "c")
        _pair(conn, "c", "d")
    with pg.begin() as conn:
        rep = lu.undo_link(conn, "ask-middle", now=NOW)
    assert rep["ok"] is True and rep["relabelled"] == {"b": "Ann"}
    with pg.begin() as conn:
        assert [_label(conn, i) for i in "abcd"] == [("Ann", False), ("Ann", True), ("Annie", True), ("Annie", False)]
    with pg.begin() as conn:
        from sqlalchemy import text

        conn.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_individual CASCADE"))
        _ind(conn, "p", "Pat", labeled_at=NOW - timedelta(days=2), ask="ask-p")
        _ind(conn, "q", "Quinn", copied=True, labeled_at=NOW - timedelta(days=1), ask="ask-r")
        _ind(conn, "r", "Quinn", labeled_at=NOW - timedelta(days=1), ask="ask-r")
        _card_link(conn, "ask-pq", "p", "q")
        _pair(conn, "q", "r")
    with pg.begin() as conn:
        assert lu.undo_link(conn, "ask-pq", now=NOW)["ok"] is True
        assert [_label(conn, i) for i in "pqr"] == [("Pat", False), ("Quinn", True), ("Quinn", False)]


@needs_pg
def test_the_cycle_applies_an_undo_the_hub_requested_and_records_a_refusal(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        for i in ("a", "b", "c", "d", "x"):
            _ind(conn, i)
        _card_link(conn, "ask-ok", "a", "b")
        _card_link(conn, "ask-stuck", "c", "d")
        _pair(conn, "c", "x", ask="n1")
        _pair(conn, "x", "d", ask="n2")
        conn.execute(text("UPDATE orion_ask SET answer_detail = answer_detail || "
                          "jsonb_build_object('undo_requested_at', 'now')"))
    with pg.begin() as conn:
        assert lu.apply_link_undos(conn, NOW) == {"links_undone": 1, "link_undos_refused": 1}
        assert lu.apply_link_undos(conn, NOW) == {"links_undone": 0, "link_undos_refused": 0}   # never retried
        d = {r[0]: r[1] for r in conn.execute(text("SELECT ask_id, answer_detail FROM orion_ask")).fetchall()}
        assert "undone_at" in d["ask-ok"]
        assert d["ask-stuck"]["undo_refused"].startswith("still joined")
        assert d["ask-stuck"]["undo_refused_detail"]["still_joined_by"] == ["c|x(ask n1)", "d|x(ask n2)"]


@needs_pg
def test_undo_script_is_a_dry_run_unless_apply(pg) -> None:
    import subprocess

    from sqlalchemy import text

    with pg.begin() as conn:
        _ind(conn, "a")
        _ind(conn, "b")
        _card_link(conn, "ask-link", "a", "b", at=datetime.now(timezone.utc) - timedelta(hours=1))
    script = REPO / "scripts" / "undo_vision_link.py"
    dry = subprocess.run([sys.executable, str(script), "ask-link", "--dsn", PG], capture_output=True, text=True)
    assert dry.returncode == 0 and "DRY RUN" in dry.stdout, dry.stderr
    with pg.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM vision_individual_pair_verdict")).scalar() == 1
    done = subprocess.run([sys.executable, str(script), "ask-link", "--dsn", PG, "--apply"], capture_output=True, text=True)
    assert done.returncode == 0 and "APPLIED" in done.stdout, done.stderr
    with pg.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM vision_individual_pair_verdict")).scalar() == 0
    listed = subprocess.run([sys.executable, str(script), "--list", "--dsn", PG], capture_output=True, text=True)
    assert "ask-link" in listed.stdout and "undone=" in listed.stdout


@needs_pg
def test_an_unexpected_error_on_one_undo_is_recorded_not_retried_forever(pg, monkeypatch) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        for i in ("a", "b", "c", "d"):
            _ind(conn, i)
        _card_link(conn, "ask-boom", "a", "b", at=NOW - timedelta(hours=2))
        _card_link(conn, "ask-ok", "c", "d")
        conn.execute(text("UPDATE orion_ask SET answer_detail = answer_detail || "
                          "jsonb_build_object('undo_requested_at', 'now')"))
    real = lu.undo_link

    def flaky(conn, ask_id, **kw):
        if ask_id == "ask-boom":
            conn.execute(text("SELECT 1/0"))
        return real(conn, ask_id, **kw)

    monkeypatch.setattr(lu, "undo_link", flaky)
    with pg.begin() as conn:
        assert lu.apply_link_undos(conn, NOW) == {"links_undone": 1, "link_undos_refused": 1}
    with pg.begin() as conn:
        d = {r[0]: r[1] for r in conn.execute(text("SELECT ask_id, answer_detail FROM orion_ask")).fetchall()}
        assert d["ask-boom"]["undo_refused"].startswith("error: ") and "undone_at" in d["ask-ok"]
