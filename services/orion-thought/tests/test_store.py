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
    """Regression guard (review finding): unlike `persist_reverie_chain`
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

        def execute(self, _stmt):
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


# --- load_latest_reverie_interpretation: Patch 3 context-seed ---------------
#
# The SQL WHERE/EXISTS clause (interpretation<>'' and chain-linkage) needs a
# real Postgres to exercise, same as this module's other read-filter
# functions (e.g. load_recent_chain_theme_events's theme_key filter). But the
# hollow re-validation is real Python logic (review finding: moved out of a
# raw SQL cast so it re-derives via SpontaneousThoughtV1.is_hollow() instead
# of trusting a stored flag that can go stale) -- fully testable here with
# real SpontaneousThoughtV1 fixtures, same construction pattern as
# test_reverie_spontaneous_thought.py.


def _grounded_thought_json(thought_id: str, interpretation: str, **overrides) -> dict:
    """A real, non-hollow SpontaneousThoughtV1 payload -- same fixture shape
    as test_reverie_spontaneous_thought.py's _coalition()/GROUNDED_TEXT
    pattern, not a hand-rolled dict that might not match the real schema's
    validation rules. `overrides` replaces (not adds to) the evidence_refs
    default -- pass evidence_refs=[...] explicitly to build a deliberately
    un-anchored/hollow fixture."""
    from orion.schemas.reverie import SpontaneousThoughtV1
    from orion.schemas.thought import CoalitionSnapshotV1

    coalition = CoalitionSnapshotV1(
        attended_node_ids=["n-1"],
        selected_open_loop_id="ol-1",
        open_loop_ids=["ol-1"],
        generated_at="2026-07-06T00:00:00Z",
    )
    fields = {
        "thought_id": thought_id,
        "correlation_id": "c",
        "coalition": coalition,
        "interpretation": interpretation,
        "evidence_refs": ["ol-1"],
        **overrides,
    }
    thought = SpontaneousThoughtV1(**fields).marked_hollow()
    return thought.model_dump(mode="json")


def _rows_result(payloads: list[dict]):
    class _FakeResult:
        def mappings(self):
            return self

        def all(self):
            return [{"thought_json": p} for p in payloads]

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, _stmt, *args):
            return _FakeResult()

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    return _FakeEngine()


def test_load_latest_reverie_interpretation_returns_grounded_candidate() -> None:
    store = _fresh_store()
    payload = _grounded_thought_json("t-1", "a real, grounded reverie thought about the mesh")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _rows_result([payload]))
    try:
        assert store.load_latest_reverie_interpretation() == (
            "a real, grounded reverie thought about the mesh"
        )
    finally:
        monkeypatch.undo()


def test_load_latest_reverie_interpretation_rejects_stale_hollow_flag() -> None:
    """Review finding: a raw SQL `thought_json->>'hollow'` cast would trust a
    stale stored flag. Build a payload where the STORED `hollow` field is
    False (as if written before a schema/guard change) but a fresh
    `is_hollow()` re-derivation says True (no coalition) -- must be
    rejected, matching chat_stance.py::_project_reverie_glimpse's own
    "gate on BOTH" discipline for this exact table."""
    from orion.schemas.reverie import SpontaneousThoughtV1

    stale = SpontaneousThoughtV1(
        thought_id="t-stale",
        correlation_id="c",
        coalition=None,  # is_hollow() -> "absent_coalition" -> True
        interpretation="a real, grounded reverie thought about the mesh",
        evidence_refs=["ol-1"],
        hollow=False,  # the stale, no-longer-trustworthy stored flag
    )
    payload = stale.model_dump(mode="json")
    assert payload["hollow"] is False  # sanity: this is the trap the fix closes

    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _rows_result([payload]))
    try:
        assert store.load_latest_reverie_interpretation() is None
    finally:
        monkeypatch.undo()


def test_load_latest_reverie_interpretation_skips_hollow_falls_through_to_next() -> None:
    """The most recent candidate is stale-hollow; the next one is real --
    the function must keep scanning, not stop at the first row."""
    store = _fresh_store()
    hollow_payload = _grounded_thought_json(
        "t-hollow", "hmm", evidence_refs=[]  # too short + unanchored -> hollow
    )
    real_payload = _grounded_thought_json("t-real", "a real, grounded reverie thought about the mesh")

    monkeypatch = pytest.MonkeyPatch()
    # ORDER BY created_at DESC -- hollow_payload is the newer (first) row.
    monkeypatch.setattr(store, "_get_engine", lambda: _rows_result([hollow_payload, real_payload]))
    try:
        assert store.load_latest_reverie_interpretation() == "a real, grounded reverie thought about the mesh"
    finally:
        monkeypatch.undo()


def test_load_latest_reverie_interpretation_skips_unparsable_row() -> None:
    """A row whose thought_json no longer validates as SpontaneousThoughtV1
    (schema drift, corruption) is skipped, not raised -- best-effort."""
    store = _fresh_store()
    real_payload = _grounded_thought_json("t-real", "a real, grounded reverie thought about the mesh")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine", lambda: _rows_result([{"not": "a valid payload"}, real_payload])
    )
    try:
        assert store.load_latest_reverie_interpretation() == "a real, grounded reverie thought about the mesh"
    finally:
        monkeypatch.undo()


def test_load_latest_reverie_interpretation_truncates_at_word_boundary() -> None:
    """Review finding: a raw slice can cut mid-word. Use real prose (not a
    uniform 'x'*N string, which can never reveal a mid-word cut) and assert
    the same truncate_at_word_boundary contract chat_history_compactor/
    github_compactor rely on -- an ellipsis, no partial trailing word."""
    store = _fresh_store()
    word = "wondering "  # 10 chars incl. space, divides evenly for a clean boundary check
    long_text = (word * 30).strip()  # well over the 240-char cap, all real words
    payload = _grounded_thought_json("t-1", long_text)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _rows_result([payload]))
    try:
        value = store.load_latest_reverie_interpretation()
    finally:
        monkeypatch.undo()

    assert value is not None
    assert len(value) <= store.MAX_REVERIE_CONTEXT_CHARS + 1  # +1 for the ellipsis char
    assert value.endswith("…")
    # The word before the ellipsis is whole ("wondering"), never a fragment
    # like "wonderin" -- what a raw `[:240]` slice would have produced here.
    assert value[:-1].endswith("wondering")


def test_load_latest_reverie_interpretation_char_limit_override() -> None:
    """char_limit=None preserves the old default (MAX_REVERIE_CONTEXT_CHARS,
    still 240); a caller-supplied value (visual_chain.py passes
    settings.reverie_context_char_limit) must actually be used, not ignored."""
    store = _fresh_store()
    long_text = ("wondering " * 30).strip()
    payload = _grounded_thought_json("t-1", long_text)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _rows_result([payload]))
    try:
        value = store.load_latest_reverie_interpretation(char_limit=20)
    finally:
        monkeypatch.undo()

    assert value is not None
    assert len(value) <= 21  # +1 for the ellipsis char, not the 240 default


def _capturing_engine(payload: dict, captured: dict):
    """Same shape as `_rows_result`, but also records the executed statement
    text and bound params -- shared by the two max_age_sec tests below
    (review finding: they previously each redefined this trio verbatim)."""

    class _FakeResult:
        def mappings(self):
            return self

        def all(self):
            return [{"thought_json": payload}]

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


def test_load_latest_reverie_interpretation_max_age_sec_adds_and_binds_the_clause() -> None:
    """Staleness bound (post-Patch-3 review finding): without an age filter,
    a stalled text-reverie worker leaves the same old thought answering
    every call forever, presented as current. Confirm the SQL actually
    carries the bound and the parameter is really passed through."""
    store = _fresh_store()
    payload = _grounded_thought_json("t-1", "a real, grounded reverie thought still fresh enough")
    captured: dict = {}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _capturing_engine(payload, captured))
    try:
        value = store.load_latest_reverie_interpretation(max_age_sec=900.0)
    finally:
        monkeypatch.undo()

    assert value == "a real, grounded reverie thought still fresh enough"
    assert "make_interval" in captured["stmt"]
    assert captured["params"]["max_age_sec"] == 900.0
    assert captured["params"]["limit"] == store._REVERIE_CONTEXT_CANDIDATE_LIMIT


def test_load_latest_reverie_interpretation_no_max_age_sec_omits_the_clause() -> None:
    store = _fresh_store()
    payload = _grounded_thought_json("t-1", "a real, grounded reverie thought about whatever is there")
    captured: dict = {}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _capturing_engine(payload, captured))
    try:
        store.load_latest_reverie_interpretation()  # max_age_sec=None, default
    finally:
        monkeypatch.undo()

    assert "make_interval" not in captured["stmt"]


def test_load_latest_reverie_interpretation_none_on_empty_table() -> None:
    store = _fresh_store()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _rows_result([]))
    try:
        assert store.load_latest_reverie_interpretation() is None
    finally:
        monkeypatch.undo()


def test_load_latest_reverie_interpretation_never_raises_on_db_failure() -> None:
    store = _fresh_store()

    class _FakeEngine:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())
    try:
        # Must degrade to None, exactly like an absent prior_description --
        # never break the visual chain's tick over a DB hiccup.
        assert store.load_latest_reverie_interpretation() is None
    finally:
        monkeypatch.undo()


# --- load_latest_self_study_reflection: Patch 5 richer context-seed --------
#
# Real incident this reader exists because of, not a hypothetical (see the
# function's own docstring): live-checking the candidate `memory_
# crystallizations` table for an "actual memory" context-seed found its
# summary/subject columns hold verbatim personal chat content.
# self_study_analysis.py's four deterministic window-contrast producers were
# the one candidate that live-verified safe (pure numeric prose, no chat
# quotes) -- confirmed by reading real bodies before writing this reader.
# The allowlist is a real privacy boundary, not a style choice, so it gets
# direct SQL-shape assertions below, not just "does it return the row" --
# asserted against `orion.schemas.self_study_analysis.ANALYSIS_SOURCES`
# itself (review finding: an earlier version of this test compared against
# a second hardcoded literal set, which couldn't catch the allowlist and the
# real producer list drifting apart).


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
        "Self-study analysis of affective state: the last 6h against the 6h "
        "before it. mean_shift: mean word_count moved 2073 vs 3.625."
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store, "_get_engine", lambda: _body_row_result({"body": body}))
    try:
        assert store.load_latest_self_study_reflection() == body
    finally:
        monkeypatch.undo()


def test_load_latest_self_study_reflection_query_only_allowlists_the_four_safe_prefixes() -> None:
    """The actual privacy boundary: assert the SQL only ever admits the four
    known-safe source_ref prefixes, by name -- not "some WHERE clause exists".
    A future edit widening this without updating the allowlist should fail
    this test, not slip through as "query still returns rows". Asserted
    against `ANALYSIS_SOURCES` itself, not a second hardcoded literal set
    (review finding: a hardcoded expectation here couldn't catch the
    allowlist and the real producer list drifting apart)."""
    from orion.schemas.self_study_analysis import ANALYSIS_SOURCES

    store = _fresh_store()
    captured: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        store, "_get_engine", lambda: _body_capturing_engine({"body": "x"}, captured)
    )
    try:
        store.load_latest_self_study_reflection()
    finally:
        monkeypatch.undo()

    stmt = captured["stmt"]
    assert "source_kind = 'self_study'" in stmt
    # starts_with(), not LIKE (review finding): every real prefix contains an
    # underscore, which LIKE treats as a single-character wildcard rather
    # than a literal -- starts_with() is exact literal prefix matching.
    assert "starts_with(source_ref" in stmt
    assert "LIKE" not in stmt
    params = captured["params"]
    bound_prefixes = {v for k, v in params.items() if k.startswith("prefix")}
    assert bound_prefixes == {f"{source}:" for source in ANALYSIS_SOURCES}
    # The real incident this allowlist closes: the free-form "Curiosity"
    # reflection's own prefix must never be one of the admitted ones.
    assert "curiosity:" not in bound_prefixes


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


# --- load_latest_memory_crystallization: Patch 6 third context-seed --------
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


def test_get_expectation_read_engine_sets_connect_timeout_too() -> None:
    """Review finding: this engine already set statement_timeout but not
    connect_timeout (a stuck TCP handshake before any query runs isn't
    bounded by a GUC statement_timeout)."""
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
        store._get_expectation_read_engine()
    finally:
        sqlalchemy.create_engine = original

    connect_args = captured.get("connect_args") or {}
    assert connect_args.get("connect_timeout") == store._EXPECTATION_ENGINE_CONNECT_TIMEOUT_SEC


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
