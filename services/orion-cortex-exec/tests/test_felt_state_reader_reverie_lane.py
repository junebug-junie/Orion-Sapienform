"""The latest_reverie_question lane in felt-state reader (reveries that ask, 2026-11-24)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.substrate_felt_state_reader import LATEST_REVERIE_QUESTION_CTX_KEY, _LANES, SubstrateFeltStateReader


def _lane(ctx_key):
    return next(lane for lane in _LANES if lane.ctx_key == ctx_key)


def _reader() -> SubstrateFeltStateReader:
    return SubstrateFeltStateReader(enabled=True, database_url="postgresql://unused/unused", max_age_sec=120)


def test_reverie_lane_reads_accepted_questions_not_the_retired_narration():
    assert LATEST_REVERIE_QUESTION_CTX_KEY == "latest_reverie_question"
    lane = _lane(LATEST_REVERIE_QUESTION_CTX_KEY)
    assert lane.table == "reverie_question"
    assert "refused_reason IS NULL" in lane.where_sql and "none_reason IS NULL" in lane.where_sql
    assert "seed_material" not in lane.payload_col  # only the question and its time cross over
    assert lane.max_age_sec == 6 * 3600 and lane.cache_ttl_sec == 120
    assert not any(l.table == "substrate_reverie_thought" for l in _LANES)


def test_hydrate_injects_a_question_from_this_afternoon(monkeypatch):
    reader = _reader()
    q = {"question_id": "q-1", "question": "Was it her, unrecognised?", "created_at": "x"}

    def fake_fetch(lane):
        if lane.ctx_key == LATEST_REVERIE_QUESTION_CTX_KEY:
            return (q, datetime.now(timezone.utc) - timedelta(hours=2))
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert ctx[LATEST_REVERIE_QUESTION_CTX_KEY] == q


def test_hydrate_rejects_a_question_older_than_six_hours(monkeypatch):
    reader = _reader()

    def fake_fetch(lane):
        if lane.ctx_key == LATEST_REVERIE_QUESTION_CTX_KEY:
            return ({"question": "old"}, datetime.now(timezone.utc) - timedelta(hours=7))
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert LATEST_REVERIE_QUESTION_CTX_KEY not in ctx


def test_hydrate_absent_row_leaves_ctx_key_unset(monkeypatch):
    reader = _reader()
    monkeypatch.setattr(reader, "_fetch_lane", lambda lane: None)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert LATEST_REVERIE_QUESTION_CTX_KEY not in ctx
