"""Tests for app.store's shared Postgres engine and startup pool warm-up.

Live-verified 2026-07-17: the first query against a freshly-created engine
pays a full TCP+auth handshake to Postgres (~400ms), enough to trip a caller
with a tight 400ms budget (formerly orion-thought's drive_state_compact
facet fetch, removed 2026-07-30) on turn one of every fresh container start.
`warm_pool()` fixes that by opening one throwaway connection at startup,
unconditionally, since every `_get_engine()` caller in this service shares
the one pool.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


def _fresh_store():
    """Reimport app.store through the module object so monkeypatch.setattr
    targets the same module object the function under test resolves its
    globals through."""
    import importlib

    import app.store as store

    importlib.reload(store)
    return store


# --- _get_engine: concurrent-construction race guard ---


def test_get_engine_returns_same_instance_across_calls() -> None:
    store = _fresh_store()

    class _FakeEngine:
        pass

    monkeypatch_created = []

    def _fake_create_engine(*_args, **_kwargs):
        engine = _FakeEngine()
        monkeypatch_created.append(engine)
        return engine

    import sqlalchemy

    original = sqlalchemy.create_engine
    sqlalchemy.create_engine = _fake_create_engine
    try:
        first = store._get_engine()
        second = store._get_engine()
    finally:
        sqlalchemy.create_engine = original

    assert first is second
    assert len(monkeypatch_created) == 1


def test_get_engine_concurrent_calls_construct_only_one_engine() -> None:
    """Two threads racing through the check-then-create-then-assign must not
    both win -- the lock added 2026-07-17 makes the second thread block until
    the first has assigned `_engine`, then reuse it instead of constructing a
    second, silently-discarded Engine/pool."""
    store = _fresh_store()
    created: list[object] = []
    entered = threading.Barrier(2, timeout=5.0)

    class _FakeEngine:
        pass

    def _fake_create_engine(*_args, **_kwargs):
        # Rendezvous both threads inside the lock-protected critical section's
        # construction call, so if the lock were absent, both would be here
        # concurrently and both would construct an engine.
        engine = _FakeEngine()
        created.append(engine)
        return engine

    import sqlalchemy

    original = sqlalchemy.create_engine
    sqlalchemy.create_engine = _fake_create_engine
    results: list[object] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            entered.wait()
        except threading.BrokenBarrierError:
            pass
        try:
            results.append(store._get_engine())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=_worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
    finally:
        sqlalchemy.create_engine = original

    assert not errors
    assert len(created) == 1, f"expected exactly one Engine constructed, got {len(created)}"
    assert results[0] is results[1]


# --- _warm_pool_sync: throwaway connection ---


def test_warm_pool_sync_executes_select_1(monkeypatch) -> None:
    store = _fresh_store()
    executed: list[str] = []

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, stmt):
            executed.append(str(stmt))

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())

    store._warm_pool_sync()
    assert any("SELECT 1" in stmt for stmt in executed)


def test_warm_pool_sync_never_raises_on_connect_failure(monkeypatch) -> None:
    store = _fresh_store()

    class _FakeEngine:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())

    # Must not raise -- a DB that isn't reachable yet at boot must not fail startup.
    store._warm_pool_sync()


# --- warm_pool: bounded async wrapper ---


@pytest.mark.asyncio
async def test_warm_pool_never_raises_on_sync_side_exception(monkeypatch) -> None:
    store = _fresh_store()

    def _boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(store, "_warm_pool_sync", _boom)

    # Must not raise even if _warm_pool_sync's own internal guard somehow
    # didn't catch it (defense-in-depth, not expected in practice).
    await store.warm_pool()


@pytest.mark.asyncio
async def test_warm_pool_outer_except_covers_non_timeout_wrapper_failure(monkeypatch) -> None:
    """Exercises the outer `except Exception` in warm_pool -- the belt-and-
    suspenders guard on top of _warm_pool_sync's own internal try/except,
    for a failure in the asyncio scaffolding itself rather than the query."""
    store = _fresh_store()

    async def _boom_to_thread(*_args, **_kwargs):
        raise RuntimeError("executor unavailable")

    monkeypatch.setattr(store.asyncio, "to_thread", _boom_to_thread)

    # Must not raise -- the outer except Exception must catch this too.
    await store.warm_pool()


# --- persist_reverie_visual_chain: chain_json column content ---------------


def test_persist_reverie_visual_chain_writes_only_its_own_chain_json_field() -> None:
    """Regression guard (review finding): unlike the retired reverie-chain writer
    (whose `ReverieChainV1` has no `chain_json` field of its own, so a full
    `model_dump()` IS the right thing to store), `ReverieVisualChainV1` has
    its own small `chain_json: dict` field. Writing the full model dump here
    self-nests the real prompt/description data one level deeper than every
    reader (including `load_latest_visual_chain_continuity_state`'s sibling
    reads and any future consumer) expects."""
    import json as _json

    from orion.schemas.reverie_visual import ReverieVisualChainV1

    store = _fresh_store()
    captured: dict = {}

    class _FakeConn:
        def execute(self, _stmt, params):
            captured.update(params)

    class _FakeBegin:
        def __enter__(self):
            return _FakeConn()

        def __exit__(self, *exc):
            return False

    class _FakeEngine:
        def begin(self):
            return _FakeBegin()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())
    try:
        chain = ReverieVisualChainV1(
            chain_id="c-1",
            terminal_reason="max_steps",
            prior_description="a warm room",
            chain_json={"prompt": "p", "artifact_sha256": "s" * 64, "description": "a warm room"},
        )
        ok = store.persist_reverie_visual_chain(chain)
    finally:
        monkeypatch.undo()

    assert ok is True
    written = _json.loads(captured["chain_json"])
    assert written == {"prompt": "p", "artifact_sha256": "s" * 64, "description": "a warm room"}
    # The bug this guards against: writing model_dump() instead would nest
    # the whole chain (including this same chain_json) one level deeper.
    assert "chain_json" not in written


# --- load_latest_visual_chain_continuity_state: Patch 4 reset counter ------
#
# Review finding: this used to be two separate functions/round trips
# (load_latest_visual_chain_prior_description, a Patch 2 original, plus a
# Patch 4 load_latest_visual_chain_continuity_streak) reading two columns of
# the SAME latest row. Consolidated into one query -- both are retired,
# nothing else in the repo called either.


def _connect_result_engine(row: dict | None):
    class _FakeResult:
        def mappings(self):
            return self

        def first(self):
            return row

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, _stmt, _params=None):
            return _FakeResult()

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    return _FakeEngine()


def test_load_latest_visual_chain_continuity_state_reads_all_three_fields() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store,
        "_get_engine",
        lambda: _connect_result_engine(
            {
                "prior_description": "an aqueduct",
                "chain_json": {"continuity_streak": 2, "context_slot_rotation": 5},
            }
        ),
    )
    try:
        assert store.load_latest_visual_chain_continuity_state() == ("an aqueduct", 2, 5)
    finally:
        monkeypatch.undo()


def test_load_latest_visual_chain_continuity_state_prior_description_empty_or_whitespace_is_none() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store,
        "_get_engine",
        lambda: _connect_result_engine({"prior_description": "   ", "chain_json": {}}),
    )
    try:
        prior, streak, rotation = store.load_latest_visual_chain_continuity_state()
    finally:
        monkeypatch.undo()
    assert prior is None
    assert streak == 0
    assert rotation == 0


def test_load_latest_visual_chain_continuity_state_streak_and_rotation_zero_on_missing_key() -> None:
    """A pre-Patch-4/pre-Patch-7 row has no continuity_streak/
    context_slot_rotation keys at all -- both degrade to 0 (the honest
    'nothing recorded yet' answer), never raise."""
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store,
        "_get_engine",
        lambda: _connect_result_engine({"prior_description": "an aqueduct", "chain_json": {"prompt": "p"}}),
    )
    try:
        assert store.load_latest_visual_chain_continuity_state() == ("an aqueduct", 0, 0)
    finally:
        monkeypatch.undo()


def test_load_latest_visual_chain_continuity_state_streak_and_rotation_zero_on_non_dict_chain_json() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store,
        "_get_engine",
        lambda: _connect_result_engine({"prior_description": None, "chain_json": "not a dict"}),
    )
    try:
        assert store.load_latest_visual_chain_continuity_state() == (None, 0, 0)
    finally:
        monkeypatch.undo()


def test_load_latest_visual_chain_continuity_state_defaults_on_empty_table() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _connect_result_engine(None))
    try:
        assert store.load_latest_visual_chain_continuity_state() == (None, 0, 0)
    finally:
        monkeypatch.undo()


def test_load_latest_visual_chain_continuity_state_never_raises_on_db_failure() -> None:
    store = _fresh_store()

    class _FakeEngine:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())
    try:
        assert store.load_latest_visual_chain_continuity_state() == (None, 0, 0)
    finally:
        monkeypatch.undo()


# --- load_latest_no_coalition_streak: text chain's own stuck-loop counter --
#
# Same shape as load_latest_visual_chain_continuity_state's streak field
# (above), but for the text reverie chain, keyed by theme_key rather than
# "the one latest row" (this service runs many themes' chains at once).












# --- load_latest_reverie_question_context: the painting's reverie seed (2026-11-24) ----------------
# The SQL filters (accepted only, age) run in Postgres (tests/test_visual_activity.py exercises them on a
# real database); these check the Python half: the shape mapping, truncation, the bound age clause, and
# fail-open. The old narrated-thought loader read substrate_reverie_thought and is deleted.


def _question_engine(row: dict | None, captured: dict):
    class _Result:
        def mappings(self):
            return self

        def first(self):
            return row

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, stmt, params=None):
            captured["sql"] = str(stmt)
            captured["params"] = dict(params or {})
            return _Result()

    class _Engine:
        def connect(self):
            return _Conn()

    return _Engine()


def _question_row(**kw):
    base = {"question_id": "q1", "thread_id": "t1", "question": "Which turns do I call short that run long?",
            "refs": ["self_bet_bin:f:0.1-0.2"], "correlation_id": "c1",
            "created_at": datetime(2026, 11, 24, 12, tzinfo=timezone.utc)}
    base.update(kw)
    return base


def test_reverie_question_context_maps_the_accepted_question() -> None:
    store = _fresh_store()
    captured: dict = {}
    with patch.object(store, "_get_engine", return_value=_question_engine(_question_row(), captured)):
        ctx = store.load_latest_reverie_question_context(max_age_sec=86400.0)
    assert ctx.text == "Which turns do I call short that run long?"
    assert ctx.thought_id == "reverie_question:q1" and ctx.text_chain_id == "reverie_thread:t1"
    assert ctx.evidence_refs == ["self_bet_bin:f:0.1-0.2"] and ctx.coalition.attended_node_ids == []
    sql = captured["sql"]
    assert "FROM reverie_question" in sql and "substrate_reverie" not in sql
    assert "refused_reason IS NULL" in sql and "none_reason IS NULL" in sql
    assert "make_interval(secs => :max_age_sec)" in sql and captured["params"]["max_age_sec"] == 86400.0


def test_reverie_question_context_truncates_and_omits_age_clause_when_unset() -> None:
    store = _fresh_store()
    captured: dict = {}
    row = _question_row(question="word " * 100)
    with patch.object(store, "_get_engine", return_value=_question_engine(row, captured)):
        ctx = store.load_latest_reverie_question_context(char_limit=20)
    assert len(ctx.text) <= 21 and "max_age_sec" not in captured["sql"]


def test_reverie_question_context_none_when_empty_and_never_raises() -> None:
    store = _fresh_store()
    with patch.object(store, "_get_engine", return_value=_question_engine(None, {})):
        assert store.load_latest_reverie_question_context() is None
    with patch.object(store, "_get_engine", side_effect=RuntimeError("db down")):
        assert store.load_latest_reverie_question_context() is None


def test_the_narrated_thought_loader_is_gone() -> None:
    store = _fresh_store()
    assert not hasattr(store, "load_latest_reverie_interpretation")
    src = Path(store.__file__).read_text()
    assert "FROM substrate_reverie_thought" not in src


# --- load_latest_self_study_reflection: Patch 5 richer context-seed --------
#
# Repointed 2026-11-07 (ladder retirement PR 6): the seed now reads curiosity's self-inquiry
# answers (self_concept_history, produced_by='curiosity_self_inquiry'). The analysis rows it read
# before had one trigger, the retired dispatch rung, and stopped on 2026-11-05.


def _body_row_result(row: dict | None):
    class _FakeResult:
        def mappings(self):
            return self

        def first(self):
            return row

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, _stmt, _params=None):
            return _FakeResult()

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    return _FakeEngine()


def _body_capturing_engine(row: dict | None, captured: dict):
    class _FakeResult:
        def mappings(self):
            return self

        def first(self):
            return row

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, stmt, params=None):
            captured["stmt"] = str(stmt)
            captured["params"] = params
            return _FakeResult()

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    return _FakeEngine()


def test_load_latest_self_study_reflection_returns_real_body() -> None:
    store = _fresh_store()
    body = (
        "Four prediction-error loops hold 97,976 of my 201,515 attention rows. "
        "I attend biometrics constantly and never ask about it."
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result({"body": body}))
    try:
        assert store.load_latest_self_study_reflection() == body
    finally:
        monkeypatch.undo()


def test_load_latest_self_study_reflection_reads_curiosity_self_inquiry_answers() -> None:
    """The seed's source is curiosity's answer of record, by producer, newest first. It never reads
    the retired analysis rows in journal_entries (their only trigger was the dispatch rung)."""
    store = _fresh_store()
    captured: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine", lambda: _body_capturing_engine({"body": "x", "source_id": "e1"}, captured)
    )
    try:
        store.load_latest_self_study_reflection()
    finally:
        monkeypatch.undo()

    stmt = " ".join(captured["stmt"].split())
    assert "FROM self_concept_history" in stmt
    assert "journal_entries" not in stmt
    assert "produced_by = :produced_by" in stmt
    assert "ORDER BY created_at DESC LIMIT 1" in stmt
    assert captured["params"]["produced_by"] == "curiosity_self_inquiry"


def test_load_latest_self_study_reflection_with_identity_names_the_answer_row() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine",
        lambda: _body_row_result({"body": "I am Orion, one presence held in four kinds of stuff.",
                                  "source_id": "sch-123"}),
    )
    try:
        seed = store.load_latest_self_study_reflection(with_identity=True)
    finally:
        monkeypatch.undo()
    assert seed.source_id == "sch-123"
    assert seed.text.startswith("I am Orion")


def test_load_latest_self_study_reflection_char_limit_override() -> None:
    store = _fresh_store()
    long_body = ("finding " * 80).strip()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result({"body": long_body}))
    try:
        value = store.load_latest_self_study_reflection(char_limit=20)
    finally:
        monkeypatch.undo()

    assert value is not None
    assert len(value) <= 21  # +1 for the ellipsis char, not the module default


def test_load_latest_self_study_reflection_max_age_sec_adds_and_binds_the_clause() -> None:
    store = _fresh_store()
    captured: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine", lambda: _body_capturing_engine({"body": "a real finding"}, captured)
    )
    try:
        value = store.load_latest_self_study_reflection(max_age_sec=21600.0)
    finally:
        monkeypatch.undo()

    assert value == "a real finding"
    assert "make_interval" in captured["stmt"]
    assert captured["params"]["max_age_sec"] == 21600.0


def test_load_latest_self_study_reflection_no_max_age_sec_omits_the_clause() -> None:
    store = _fresh_store()
    captured: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine", lambda: _body_capturing_engine({"body": "a real finding"}, captured)
    )
    try:
        store.load_latest_self_study_reflection()  # max_age_sec=None, default
    finally:
        monkeypatch.undo()

    assert "make_interval" not in captured["stmt"]


def test_load_latest_self_study_reflection_none_on_empty_table() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result(None))
    try:
        assert store.load_latest_self_study_reflection() is None
    finally:
        monkeypatch.undo()


def test_load_latest_self_study_reflection_none_on_empty_body() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result({"body": "   "}))
    try:
        assert store.load_latest_self_study_reflection() is None
    finally:
        monkeypatch.undo()


def test_load_latest_self_study_reflection_never_raises_on_db_failure() -> None:
    store = _fresh_store()

    class _FakeEngine:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())
    try:
        assert store.load_latest_self_study_reflection() is None
    finally:
        monkeypatch.undo()


# --- load_episode_memory_seed: Stage 2 PR G visual-memory source -----------


def test_load_episode_memory_seed_returns_typed_source_identity() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store,
        "_get_engine",
        lambda: _body_row_result(
            {"source_id": "a47eb72a-7010-4ea4-a90d-f025166ac2fd", "statement": "A kept memory."}
        ),
    )
    try:
        source = store.load_episode_memory_seed(with_identity=True)
    finally:
        monkeypatch.undo()

    assert source.source_id == "a47eb72a-7010-4ea4-a90d-f025166ac2fd"
    assert source.text == "A kept memory."


def test_load_episode_memory_seed_query_enforces_governance_and_rotation() -> None:
    """The SQL is the privacy and diversity boundary; lock its joins and gates."""
    store = _fresh_store()
    captured: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store,
        "_get_engine",
        lambda: _body_capturing_engine(
            {"source_id": "a47eb72a-7010-4ea4-a90d-f025166ac2fd", "statement": "x"}, captured
        ),
    )
    try:
        store.load_episode_memory_seed(max_age_sec=1209600.0)
    finally:
        monkeypatch.undo()

    stmt = captured["stmt"]
    assert "m.purpose IN ('happened', 'about_juniper')" in stmt
    assert "m.status = 'active'" in stmt
    assert "m.expires_at IS NULL OR m.expires_at > now()" in stmt
    assert "m.stakes = 'low' AND m.confirmation_state = 'auto'" in stmt
    assert "m.confirmation_state = 'confirmed'" in stmt
    assert "ev.memory_id = m.memory_id AND ev.verified IS TRUE" in stmt
    assert "ev.memory_id = m.memory_id AND ev.verified IS NOT TRUE" in stmt
    assert "e.memory_id = m.memory_id" in stmt
    assert "e.op IN ('downgraded_voice', 'rejected_invalid')" in stmt
    assert "source_kind' = 'episode_memory'" in stmt
    assert "s.last_seeded_at ASC NULLS FIRST" in stmt
    assert "e.created_at ASC" in stmt
    assert captured["params"] == {"max_age_sec": 604800.0}


def test_load_episode_memory_seed_truncates_at_word_boundary() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store,
        "_get_engine",
        lambda: _body_row_result({"source_id": "memory-id", "statement": "shared memory " * 30}),
    )
    try:
        value = store.load_episode_memory_seed(char_limit=20)
    finally:
        monkeypatch.undo()

    assert value is not None
    assert len(value) <= 21


def test_load_episode_memory_seed_none_on_empty_or_db_failure() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result(None))
    try:
        assert store.load_episode_memory_seed() is None
    finally:
        monkeypatch.undo()

    class _FakeEngine:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())
    try:
        assert store.load_episode_memory_seed() is None
    finally:
        monkeypatch.undo()


# --- load_latest_memory_crystallization: Patch 6 legacy context-seed -------
#
# Reverses Patch 5's declined call on this same table -- see the function's
# own docstring for why. Real correction (design doc §20): `status='active'`
# alone is NOT a governor-review signal for most of this table -- verified
# live that `formation_policy.py`'s AUTO_ACTIVE_KINDS sets it on creation
# with zero review. The actual filter requires a real `memory_
# crystallization_history` row with `op='approve'` -- asserted directly
# below so a future edit that drops that requirement silently fails this
# test, not just "query still returns a row".


def test_load_latest_memory_crystallization_returns_real_summary() -> None:
    store = _fresh_store()
    summary = "Orion and Juniper talked through the week's mesh work."
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result({"summary": summary}))
    try:
        assert store.load_latest_memory_crystallization() == summary
    finally:
        monkeypatch.undo()


def test_load_latest_memory_crystallization_query_requires_active_status_and_real_approval() -> None:
    """The actual privacy/governance boundary: `status='active'` alone
    proved (live, design doc §20) to be a near-universal default most rows
    get with zero review -- the query must also require a real
    `memory_crystallization_history` row with `op='approve'`, not just
    `status='active'`."""
    store = _fresh_store()
    captured: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine", lambda: _body_capturing_engine({"summary": "x"}, captured)
    )
    try:
        store.load_latest_memory_crystallization()
    finally:
        monkeypatch.undo()

    stmt = captured["stmt"]
    assert "memory_crystallizations" in stmt
    assert "status = 'active'" in stmt
    # Review finding: substring-checking "memory_crystallization_history" and
    # "op = 'approve'" separately would still pass if a future edit
    # decorrelated the EXISTS subquery (e.g. dropped the crystallization_id
    # join predicate, leaving "is there ANY approved row anywhere in the
    # whole table" -- true almost always, since real approved rows already
    # exist -- instead of "is THIS row approved"). Assert the actual join
    # predicate is present, not just that both fragments appear somewhere.
    assert "h.crystallization_id = mc.crystallization_id" in stmt
    assert "h.op = 'approve'" in stmt
    # Deliberately no source_ref/allowlist clause -- unlike self-study, this
    # reader is not content-filtered (see the function's own docstring).
    assert "source_ref" not in stmt


def test_load_latest_memory_crystallization_char_limit_override() -> None:
    store = _fresh_store()
    long_summary = ("shared memory " * 80).strip()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result({"summary": long_summary}))
    try:
        value = store.load_latest_memory_crystallization(char_limit=20)
    finally:
        monkeypatch.undo()

    assert value is not None
    assert len(value) <= 21  # +1 for the ellipsis char, not the module default


def test_load_latest_memory_crystallization_max_age_sec_adds_and_binds_the_clause() -> None:
    store = _fresh_store()
    captured: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine", lambda: _body_capturing_engine({"summary": "a real memory"}, captured)
    )
    try:
        value = store.load_latest_memory_crystallization(max_age_sec=21600.0)
    finally:
        monkeypatch.undo()

    assert value == "a real memory"
    assert "make_interval" in captured["stmt"]
    assert captured["params"]["max_age_sec"] == 21600.0


def test_load_latest_memory_crystallization_no_max_age_sec_omits_the_clause() -> None:
    store = _fresh_store()
    captured: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine", lambda: _body_capturing_engine({"summary": "a real memory"}, captured)
    )
    try:
        store.load_latest_memory_crystallization()  # max_age_sec=None, default
    finally:
        monkeypatch.undo()

    assert "make_interval" not in captured["stmt"]


def test_load_latest_memory_crystallization_none_on_empty_table() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result(None))
    try:
        assert store.load_latest_memory_crystallization() is None
    finally:
        monkeypatch.undo()


def test_load_latest_memory_crystallization_none_on_empty_summary() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result({"summary": "   "}))
    try:
        assert store.load_latest_memory_crystallization() is None
    finally:
        monkeypatch.undo()


def test_load_latest_memory_crystallization_never_raises_on_db_failure() -> None:
    store = _fresh_store()

    class _FakeEngine:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())
    try:
        assert store.load_latest_memory_crystallization() is None
    finally:
        monkeypatch.undo()


# --- _get_engine: statement_timeout/connect_timeout (2026-09-04 wedge fix) ---
# A single stuck DB call run via asyncio.to_thread can never be cancelled by
# Python, only abandoned -- it permanently occupies one worker in the
# process-wide thread pool. Confirmed live: this is how persist_reverie_
# visual_chain wedged visual_chain.py's worker for 24+ hours with zero errors
# logged. These constants/connect_args must actually reach create_engine().


def test_get_engine_sets_statement_timeout_and_connect_timeout() -> None:
    store = _fresh_store()
    captured = {}

    class _FakeEngine:
        pass

    def _fake_create_engine(*_args, **kwargs):
        captured.update(kwargs)
        return _FakeEngine()

    import sqlalchemy

    original = sqlalchemy.create_engine
    sqlalchemy.create_engine = _fake_create_engine
    try:
        store._get_engine()
    finally:
        sqlalchemy.create_engine = original

    connect_args = captured.get("connect_args") or {}
    assert connect_args.get("connect_timeout") == store._ENGINE_CONNECT_TIMEOUT_SEC
    assert (
        f"statement_timeout={store._ENGINE_STATEMENT_TIMEOUT_MS}"
        in connect_args.get("options", "")
    )




# --- visual_chain_age_minutes ---


def _age_row_result(value):
    class _FakeResult:
        def first(self):
            return None if value is None else (value,)

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, _stmt, _params=None):
            return _FakeResult()

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    return _FakeEngine()


def test_visual_chain_age_minutes_returns_value() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _age_row_result(12.5))
    try:
        assert store.visual_chain_age_minutes() == 12.5
    finally:
        monkeypatch.undo()


def test_visual_chain_age_minutes_none_on_empty_table() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _age_row_result(None))
    try:
        assert store.visual_chain_age_minutes() is None
    finally:
        monkeypatch.undo()


def test_visual_chain_age_minutes_never_raises_on_db_failure() -> None:
    store = _fresh_store()

    class _FakeEngine:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())
    try:
        assert store.visual_chain_age_minutes() is None
    finally:
        monkeypatch.undo()


# --- resonance_alert_cooldown_active / _mark ---
#
# Real dedupe for ResonanceHealthMonitor's "worsening" pages: NotificationRequest.
# dedupe_key/dedupe_window_seconds are stored by orion-notify but never read
# back to suppress a repeat (see resonance_alert_cooldown_active's own
# docstring). This durable per-key timestamp is the actual gate.


def _cooldown_read_engine(row: dict | None):
    class _FakeResult:
        def mappings(self):
            return self

        def first(self):
            return row

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, _stmt, _params=None):
            return _FakeResult()

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    return _FakeEngine()


def _cooldown_write_engine(captured: dict):
    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, stmt, params=None):
            captured["stmt"] = str(stmt)
            captured["params"] = params

    class _FakeEngine:
        def begin(self):
            return _FakeConn()

    return _FakeEngine()
















def test_visual_context_preserves_full_producer_evidence_bound():
    from orion.schemas.reverie import MAX_EVIDENCE_REFS
    from orion.schemas.reverie_visual import ReverieVisualContextV1
    refs = [f"node-{i}" for i in range(MAX_EVIDENCE_REFS)]
    context = ReverieVisualContextV1(text="grounded context", thought_id="thought",
        thought_correlation_id="correlation", thought_created_at="2026-09-13T00:00:00Z",
        text_chain_id="settled", coalition={"attended_node_ids": refs, "open_loop_ids": [], "selected_open_loop_id": None,
        "generated_at": "2026-09-13T00:00:00Z"}, evidence_refs=refs)
    assert context.evidence_refs == refs
