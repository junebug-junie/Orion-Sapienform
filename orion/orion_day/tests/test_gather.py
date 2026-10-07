"""gather_orion_day over a fixture day: every source, the blind rule, stable compactor ids,
failure isolation, and the half-open Denver window."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from orion.cognition.chat_history_compactor.digest import stable_chat_compactor_journal_entry_id
from orion.cognition.compactor.calendar_day import previous_local_day_window
from orion.cognition.github_compactor.digest import stable_github_compactor_journal_entry_id
from orion.dream import introspect_sql
from orion.orion_day import gather
from orion.orion_day.tests import fixtures as fx
from orion.orion_day.window import orion_day_window, yesterday_letter_date
from orion.schemas.orion_day import DreamHypothesisV1


def _gather(conn):
    return asyncio.run(gather.gather_orion_day(conn, fx.LETTER_DATE, now=datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)))


def test_every_source_is_read_at_full_length():
    material = _gather(fx.FakeConn())
    assert [r.run_id for r in material.curiosity_runs] == ["ab61e4ccd47b", "f218860792b4", "nojournal0001"]
    investigate = material.curiosity_runs[0]
    assert investigate.journal_body == fx.CURIOSITY_JOURNALS[0]["body"]  # full, not the capped finding
    assert investigate.finding_text is None
    assert investigate.outcome["per_prior"][0]["prior_id"] == "self:hop_written_at_missing"
    assert material.curiosity_runs[1].self_definition_text == "I am one inference process on one substrate."
    assert material.curiosity_runs[2].finding_text == "Only the capped finding survived."
    assert [f.run_id for f in material.curiosity_failed] == ["failed000001"]
    assert material.self_sense[0].answer_text.startswith("A persistent mind")
    assert material.readings[0].learned == ("GGUF quants load through transformers now. " * 60).strip()
    assert len(material.readings[0].learned) > 900  # world_pulse_read's default cap does not apply here
    assert material.reading_journals[0].entry_id == "8c62d21d"
    assert material.dream_narratives[0].themes == ["archive"]
    assert material.dream_hypotheses[0].hypothesis_id == "dh-1a69e4d982ac"
    assert len(material.reverie_thoughts) == len(fx.REVERIE_THOUGHTS)  # every thought kept in material
    assert material.reverie_thoughts[0].interpretation == fx.REVERIE_THOUGHTS[0]["interpretation"]
    assert len(material.reverie_chains) == 100
    assert material.visual_reveries[0].path.endswith("a550.png")
    assert material.chat_compactor.entry_id == "d5f4c123"
    assert material.github_compactor is None
    assert [i.title for i in material.world_pulse_digest.items] == ["Fuel standards rolled back"]
    assert material.sources["github_compactor"].status == "empty"
    assert material.sources["reverie_thoughts"].count == len(fx.REVERIE_THOUGHTS)
    assert all(s.status in ("ok", "empty") for s in material.sources.values())


def test_self_sense_run_is_not_listed_as_a_curiosity_run():
    material = _gather(fx.FakeConn())
    assert "selfsense0001" not in {r.run_id for r in material.curiosity_runs}


def test_one_failing_source_names_its_gap_and_the_rest_still_gathers():
    material = _gather(fx.FakeConn(fail={"reverie_thoughts", "world_pulse_digest"}))
    assert material.sources["reverie_thoughts"].status == "error"
    assert "does not exist" in material.sources["reverie_thoughts"].error
    assert material.reverie_thoughts == []
    assert material.world_pulse_digest is None and material.sources["world_pulse_digest"].status == "error"
    assert material.curiosity_runs and material.readings  # untouched


def test_every_window_query_gets_the_half_open_denver_day():
    conn = fx.FakeConn()
    _gather(conn)
    window = orion_day_window(fx.LETTER_DATE)
    windowed = [args for sql, args in conn.calls if sql not in (
        gather.CURIOSITY_OUTCOMES_SQL, gather.WORLD_PULSE_DIGEST_SQL, gather.JOURNAL_BY_ID_OR_REF_SQL)]
    assert windowed and all(args[0] == window.window_start and args[1] == window.window_end for args in windowed)
    assert window.window_start == datetime(2026, 9, 29, 6, 0, tzinfo=timezone.utc)
    assert window.window_end == datetime(2026, 9, 30, 6, 0, tzinfo=timezone.utc)


def test_compactors_are_fetched_by_their_stable_ids_and_refs():
    conn = fx.FakeConn(github={**fx.CHAT_COMPACTOR, "entry_id": "gh1", "source_ref": "github_compactor_pass:x"})
    material = _gather(conn)
    lookups = [args for sql, args in conn.calls if sql == gather.JOURNAL_BY_ID_OR_REF_SQL]
    day_end = orion_day_window(fx.LETTER_DATE).window_end
    assert lookups[0] == (
        stable_chat_compactor_journal_entry_id(workflow_id="chat_history_compactor_pass",
                                               compactor_index="chat_compactor:day:2026-09-29"),
        "chat_history_compactor_pass:chat_compactor:day:2026-09-29",
        day_end,
    )
    assert lookups[1] == (
        stable_github_compactor_journal_entry_id(workflow_id="github_compactor_pass", calendar_date="2026-09-29",
                                                 repo="junebug-junie/Orion-Sapienform"),
        "github_compactor_pass:2026-09-29:junebug-junie/Orion-Sapienform",
        day_end,
    )
    assert material.github_compactor.entry_id == "gh1"


def test_a_compactor_row_written_during_the_day_is_not_the_day_digest():
    # GitHub rolling mode labels a run with its own date: a digest made DURING the day shares the
    # day digest's stable id but covers the previous 24 h.
    during = {**fx.CHAT_COMPACTOR, "entry_id": "gh-rolling", "created_at": fx.T0}
    material = _gather(fx.FakeConn(github=during))
    assert material.github_compactor is None
    assert "created_at >= $3" in gather.JOURNAL_BY_ID_OR_REF_SQL


def test_stable_compactor_ids_match_live_rows():
    # Live journal_entries rows checked 2026-09-30.
    assert stable_chat_compactor_journal_entry_id(
        workflow_id=gather.CHAT_COMPACTOR_WORKFLOW_ID, compactor_index="chat_compactor:day:2026-09-28",
    ) == "d5f4c123-ce44-5591-9e72-112a6e8f5831"
    assert stable_github_compactor_journal_entry_id(
        workflow_id=gather.GITHUB_COMPACTOR_WORKFLOW_ID, calendar_date="2026-09-27", repo=gather.DEFAULT_GITHUB_REPO,
    ) == "1d92aa5d-57d9-5485-ae02-b457674ae540"


# --- the blind rule --------------------------------------------------------------------------


def _selected_columns(sql: str) -> set[str]:
    select = sql.split("FROM", 1)[0]
    return {c.strip().split(".")[-1].split(" ")[0] for c in select.replace("SELECT", "").split(",")}


@pytest.mark.parametrize("sql", [introspect_sql.HYPOTHESIS_WINDOW_SQL, introspect_sql.HYPOTHESIS_RECENT_SQL,
                                 gather.DREAM_HYPOTHESES_SEEN_SQL])
def test_hypothesis_sql_only_reads_offered_rows_and_never_the_arm(sql):
    assert "h.offered_at IS NOT NULL" in sql
    columns = _selected_columns(sql)
    for forbidden in introspect_sql.BLIND_FORBIDDEN_COLUMNS:
        assert forbidden not in columns
        assert not re.search(rf"\b{forbidden}\b", sql.split("FROM", 1)[0])


def test_no_gather_statement_touches_the_blind_columns():
    statements = [v for k, v in vars(gather).items() if k.endswith("_SQL")]
    statements += [introspect_sql.NARRATIVE_WINDOW_SQL, introspect_sql.HYPOTHESIS_WINDOW_SQL,
                   introspect_sql.HYPOTHESIS_RECENT_SQL, introspect_sql.NARRATIVE_RECENT_SQL]
    for sql in statements:
        for forbidden in ("arm", "ref_a", "ref_b", "cycle_json"):
            assert not re.search(rf"\b{forbidden}\b", sql), (forbidden, sql[:80])


def test_letter_shows_only_hypotheses_whose_offering_run_completed():
    sql = gather.DREAM_HYPOTHESES_SEEN_SQL
    assert "h.offered_run_id" in sql and "s.status = 'completed'" in sql


def test_hypothesis_model_rejects_an_arm_field():
    with pytest.raises(ValidationError):
        DreamHypothesisV1(hypothesis_id="dh-1", claim="c", offered_at=fx.T0, arm="dream")


def test_every_statement_is_a_select():
    statements = [v for k, v in vars(gather).items() if k.endswith("_SQL")]
    for sql in statements:
        assert sql.lstrip().upper().startswith("SELECT"), sql[:60]
        assert not re.search(r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER)\b", sql.upper())


# --- the window -------------------------------------------------------------------------------


@pytest.mark.parametrize("letter_date,hours", [
    (date(2026, 3, 8), 23),   # spring forward
    (date(2026, 11, 1), 25),  # fall back
    (date(2026, 9, 29), 24),
])
def test_window_is_dst_safe(letter_date, hours):
    window = orion_day_window(letter_date)
    assert (window.window_end - window.window_start).total_seconds() == hours * 3600


def test_letter_date_matches_the_compactor_day():
    now = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)  # 08:30 Denver
    assert yesterday_letter_date(now).isoformat() == previous_local_day_window(now).calendar_date == "2026-09-29"
    compactor = previous_local_day_window(now)
    window = orion_day_window(yesterday_letter_date(now))
    assert window.window_start == compactor.window_start
    assert compactor.window_end < window.window_end  # inclusive 23:59:59.999999 vs exclusive next midnight
