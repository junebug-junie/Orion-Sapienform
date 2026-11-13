"""Camera asks as graded acts (attention patch 4a), end to end against real Postgres.

The lifecycle: an ask Orion made opens a vision.ask act; Juniper's answer closes it 1; "I don't know"
closes it 0; seven days of silence close it 0; an ask Orion withdraws closes ``expired`` (not a grade);
her own walkway-story naming card opens nothing. Then the want's per-rule outcome reads those grades.

Opt-in like the other act-ledger tests: ORION_SQL_WRITER_TEST_DATABASE_URI is a scratch database; every
table lives in a fresh throwaway schema. Row shapes copied from live orion_ask (read-only, 2026-11-13).
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ask_acts import open_ask_acts  # noqa: E402
from app.ask_event_wants import GRADES_SQL  # noqa: E402
from app.db import Base  # noqa: E402
from app.models.grammar_trace import GrammarActSQL, GrammarEventSQL, GrammarTraceSQL  # noqa: E402
from orion.grammar.act_ledger import resolve_ask_acts, resolve_due_acts  # noqa: E402
from orion.grammar.camera_ask_acts import ask_act_ids, rule_from_test_before, rule_outcome_values  # noqa: E402
from test_act_ledger_integration import L10_DDL  # noqa: E402

pytestmark = pytest.mark.integration

# The live columns of orion_ask (\\d orion_ask on conjourney, 2026-11-13) and the opener's unique index.
ASK_DDL = """
CREATE TABLE orion_ask (
    ask_id TEXT PRIMARY KEY, asked_of TEXT NOT NULL DEFAULT 'juniper', question TEXT NOT NULL,
    evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb, image_ref TEXT, status TEXT NOT NULL DEFAULT 'open',
    answer TEXT, answered_at TIMESTAMPTZ, applied_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ, source_kind TEXT NOT NULL, source_ref TEXT NOT NULL,
    image_refs JSONB NOT NULL DEFAULT '[]'::jsonb, answer_detail JSONB);
CREATE UNIQUE INDEX orion_ask_open_source_uidx ON orion_ask (source_kind, source_ref) WHERE status = 'open';
CREATE TABLE vision_individual (individual_id TEXT PRIMARY KEY, kind TEXT NOT NULL)
"""

NOW = datetime.now(timezone.utc).replace(microsecond=0)


@pytest.fixture
def sess():
    uri = os.environ["ORION_SQL_WRITER_TEST_DATABASE_URI"]
    schema = f"ask_act_test_{uuid.uuid4().hex[:10]}"
    admin = create_engine(uri)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(uri, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for stmt in (L10_DDL + ";" + ASK_DDL).split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    Base.metadata.create_all(bind=engine, tables=[GrammarActSQL.__table__, GrammarEventSQL.__table__,
                                                  GrammarTraceSQL.__table__])
    s = sessionmaker(bind=engine)()
    s.info["dsn"] = f"{uri}{'&' if '?' in uri else '?'}options=-csearch_path%3D{schema}"
    try:
        yield s
    finally:
        s.rollback()
        s.close()
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def _ask(s, ask_id, kind, ref, question, refs, *, created, status="open", answer=None, detail=None, answered=None,
         expires="default"):
    s.execute(text(
        "INSERT INTO orion_ask (ask_id, question, evidence_refs, status, answer, answered_at, created_at, expires_at, "
        "source_kind, source_ref, answer_detail) VALUES (:a, :q, CAST(:e AS jsonb), :st, :an, :at, :c, :x, :k, :r, "
        "CAST(:d AS jsonb))"),
        {"a": ask_id, "q": question, "e": json.dumps(refs), "st": status, "an": answer, "at": answered, "c": created,
         "x": created + timedelta(days=7) if expires == "default" else expires, "k": kind, "r": ref,
         "d": None if detail is None else json.dumps(detail)})


def _seed(s):
    s.execute(text("INSERT INTO vision_individual VALUES ('veh1', 'vehicle'), ('p1', 'person'), ('p2', 'person')"))
    t = NOW - timedelta(hours=2)
    _ask(s, "ask-confirm", "vision_person_confirm", "p1|Teddy",
         "Is this Teddy? It is my best guess, but I am not sure. I have seen them 9 times over 4 days.",
         ["sighting:5625", "sighting:a121", "sighting:2ae8"], created=t)
    _ask(s, "ask-vehicle", "vision_individual", "veh1",
         "I have seen this same vehicle 30 times over 5 days, usually around 07:40.",
         ["sighting:bc2c", "sighting:6a71", "sighting:be68"], created=t)
    _ask(s, "ask-same", "vision_same_individual", "p1|p2",
         "Are these the same person? Their faces look alike to me, but I am not sure.",
         ["individual:p1", "individual:p2", "face_cosine:0.558"], created=t)
    # Juniper's own cards: a new one with the marker, and a legacy one with only answer_detail.story_id.
    _ask(s, "ask-juniper-new", "vision_individual", "p2", "Who is this? (named on a walkway story)",
         ["motion_story:0255", "individual:p2", "rule:juniper_story_naming"], created=t, status="answered",
         answer="Amazon delivery person", answered=t, expires=None,
         detail={"story_id": "0255", "action": "name", "via": "motion_story", "name": "Amazon delivery person"})
    _ask(s, "ask-juniper-legacy", "vision_individual", "p1", "Who is this? (named on a walkway story)",
         ["motion_story:0256", "individual:p1"], created=t, status="answered", answer="Teddy", answered=t, expires=None,
         detail={"story_id": "0256", "action": "name", "via": "motion_story", "name": "Teddy"})
    # The marker alone also excludes a card (each clause of the filter must stand on its own).
    _ask(s, "ask-juniper-marker-only", "vision_individual", "veh1", "Who is this? (named on a walkway story)",
         ["motion_story:0257", "individual:veh1", "rule:juniper_story_naming"], created=t, status="answered",
         answer="Mail truck", answered=t, expires=None, detail={"name": "Mail truck"})
    # Not a camera ask.
    _ask(s, "ask-memory", "memory_confirmation", "loop:1", "Is this still true?", [], created=t)
    # A camera ask older than the live loop's 24 h: the back-fill's, not the loop's.
    _ask(s, "ask-old", "vision_person_confirm", "p2|Addy", "Is this Addy? It is my best guess, but I am not sure.",
         ["sighting:1"], created=NOW - timedelta(days=3), status="answered", answer="yes",
         answered=NOW - timedelta(days=3) + timedelta(hours=4))
    s.commit()


def _act(s, ask_id):
    return s.execute(text("SELECT * FROM grammar_acts WHERE act_id = :a"),
                     {"a": ask_act_ids([ask_id])[ask_id]}).mappings().first()


def test_lifecycle_open_answer_dont_know_expiry_and_juniper_cards(sess):
    _seed(sess)
    opened = open_ask_acts(sess, now=NOW)
    sess.commit()
    assert sorted(opened) == sorted(ask_act_ids(["ask-confirm", "ask-vehicle", "ask-same"]).values())
    for ask_id, rule in (("ask-confirm", "confirm"), ("ask-vehicle", "vehicle"), ("ask-same", "same")):
        a = _act(sess, ask_id)
        assert (a["actor"], a["verb"], a["signal_id"], a["object_ref"], a["correlation_id"]) == (
            "vision.ask", "ask", "juniper_answered", f"orion_ask:{ask_id}", ask_id)
        assert rule_from_test_before(a["test_before"]) == rule
        assert a["opened_at"] == NOW - timedelta(hours=2) and a["horizon_at"] == NOW - timedelta(hours=2) + timedelta(days=7)
    # Juniper's cards, the memory card and the old ask opened nothing.
    for ask_id in ("ask-juniper-new", "ask-juniper-legacy", "ask-juniper-marker-only", "ask-memory", "ask-old"):
        assert _act(sess, ask_id) is None
    assert open_ask_acts(sess, now=NOW) == []     # idempotent

    # Nothing answered yet: the resolver leaves them open.
    assert resolve_due_acts(sess, now=NOW, commit=sess.commit).closed_asks_graded == 0

    # Juniper answers "no" on the confirm card (an answer: 1) and "I don't know" on the vehicle (0).
    sess.execute(text("UPDATE orion_ask SET status='answered', answer='no', answered_at=:t WHERE ask_id='ask-confirm'"),
                 {"t": NOW - timedelta(minutes=30)})
    sess.execute(text("UPDATE orion_ask SET status='dismissed', answer_detail='{\"dont_know\": true}'::jsonb, "
                      "answered_at=:t WHERE ask_id='ask-vehicle'"), {"t": NOW - timedelta(minutes=20)})
    sess.commit()
    summary = resolve_due_acts(sess, now=NOW, commit=sess.commit)
    assert (summary.closed_asks_graded, summary.closed_asks_withdrawn) == (2, 0)
    c, v = _act(sess, "ask-confirm"), _act(sess, "ask-vehicle")
    assert (c["outcome_value"], c["outcome_source"], c["closed_at"]) == (1.0, "resolver", NOW - timedelta(minutes=30))
    assert (v["outcome_value"], v["outcome_source"]) == (0.0, "resolver") and "does not know" in v["outcome_note"]
    graded = sess.execute(text("SELECT count(*) FROM substrate_action_outcomes WHERE target_id='vision.ask' "
                               "AND signal_id='juniper_answered'")).scalar()
    assert graded == 2
    assert _act(sess, "ask-same")["closed_at"] is None

    # Seven days of silence: the same-person ask closes 0 at its expiry, graded by the resolver.
    later = NOW - timedelta(hours=2) + timedelta(days=7, minutes=1)
    summary = resolve_due_acts(sess, now=later, commit=sess.commit)
    s_act = _act(sess, "ask-same")
    assert summary.closed_asks_graded == 1
    assert (s_act["outcome_value"], s_act["outcome_source"], s_act["closed_at"]) == (
        0.0, "resolver", NOW - timedelta(hours=2) + timedelta(days=7))

    # The want's per-rule expected outcome reads exactly these grades.
    rows = sess.execute(text(GRADES_SQL), {"actor": "vision.ask", "since": NOW - timedelta(days=30)}).fetchall()
    v = rule_outcome_values((rule_from_test_before(r[0]) or "", float(r[1])) for r in rows)
    assert (v["confirm"], v["vehicle"], v["same"], v["face"]) == (2 / 3, 1 / 3, 1 / 3, 0.5)


def test_withdrawn_by_orion_is_expired_not_graded_and_the_backfill_opens_older_asks(sess):
    _seed(sess)
    open_ask_acts(sess, now=NOW)
    # A split undo expires the open same-person ask long before its expiry.
    sess.execute(text("UPDATE orion_ask SET status='expired' WHERE ask_id='ask-same'"))
    sess.commit()
    summary = resolve_due_acts(sess, now=NOW, commit=sess.commit)
    assert summary.closed_asks_withdrawn == 1
    a = _act(sess, "ask-same")
    assert (a["outcome_value"], a["outcome_source"]) == (0.0, "expired")
    assert sess.execute(text("SELECT count(*) FROM substrate_action_outcomes")).scalar() == 1   # recorded, not graded
    rows = sess.execute(text(GRADES_SQL), {"actor": "vision.ask", "since": NOW - timedelta(days=30)}).fetchall()
    assert rows == []                                  # not a grade: the expected outcome ignores it

    # The back-fill's path: open the 30-day window, then grade just those acts.
    opened = open_ask_acts(sess, now=NOW, since=NOW - timedelta(days=30))
    assert opened == [ask_act_ids(["ask-old"])["ask-old"]]
    graded, withdrawn = resolve_ask_acts(sess, opened, now=NOW, commit=sess.commit)
    old = _act(sess, "ask-old")
    assert (graded, withdrawn) == (1, 0)
    assert (old["outcome_value"], old["outcome_source"]) == (1.0, "resolver")
    assert old["opened_at"] == NOW - timedelta(days=3)


def test_backfill_script_dry_run_reads_only_then_apply_opens_and_grades(sess, tmp_path):
    """scripts/backfill_camera_ask_acts.py (CLAUDE.md section 14): dry run writes nothing to the database."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "backfill_camera_ask_acts", Path(__file__).resolve().parents[1] / "scripts" / "backfill_camera_ask_acts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _seed(sess)
    job = tmp_path / "job"
    assert mod.main(["--dsn", sess.info["dsn"], "--job-dir", str(job)]) == 0
    dry = json.loads((job / "dry_run.json").read_text())
    assert (dry["mode"], dry["targets"], dry["would_grade_1"], dry["stays_open"]) == ("dry_run", 4, 1, 3)
    assert dry["by_rule"] == {"confirm": 2, "vehicle": 1, "same": 1}
    assert (job / "snapshot.jsonl").read_text().count("\n") == 4
    assert sess.execute(text("SELECT count(*) FROM grammar_acts")).scalar() == 0      # read only
    assert mod.main(["--dsn", sess.info["dsn"], "--job-dir", str(job), "--apply"]) == 0
    sess.commit()
    assert sess.execute(text("SELECT count(*) FROM grammar_acts WHERE actor='vision.ask'")).scalar() == 4
    assert (job / "report.md").exists() and (job / "before_after.csv").read_text().count("\n") == 5
    assert "DELETE FROM grammar_acts" in (job / "report.md").read_text()
    assert _act(sess, "ask-old")["outcome_value"] == 1.0
    # A second run finds nothing to do.
    assert mod.main(["--dsn", sess.info["dsn"], "--job-dir", str(tmp_path / "again")]) == 0
    assert json.loads((tmp_path / "again" / "dry_run.json").read_text())["targets"] == 0
