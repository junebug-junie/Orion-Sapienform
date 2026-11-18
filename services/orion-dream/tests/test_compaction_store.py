"""Dream compaction downscales memory under Juniper's standing consent (2026-11-13).

Lifecycle, on a real SQL table shaped like ``substrate_reverie_thought`` (SQLite, same columns the store
touches): live-shaped REM requests and reverie salience -> the REM delta proposes downscale entries -> the
want market's pure core answers with the SHIPPED consent.yaml (allow, standing consent) -> the applier
snapshots -> the store lowers salience -> restore from the snapshot file puts every value back.

The fixture rows are real: request and thought ids, themes and salience values copied READ-ONLY from live
Postgres on 2026-11-13 (``dream_compaction_request_queue`` and ``substrate_reverie_thought``). No reverie
text is copied.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from test_compaction_consent import GRANTED, NO_STANDING, SHIPPED, consent_for  # noqa: E402

# Live, 2026-11-13: four settled-chain requests and their thoughts' salience (column == thought_json).
LIVE_REQUESTS = [
    {"theme": "open-loop-4409448b6975", "reason": "reverie_chain_settled", "op_hint": "consolidate",
     "request_id": "compaction-request:6a9ef1b2-0dcb-4b22-817f-24d4a03673af",
     "evidence_refs": ["5d39c104-bdf5-4ec7-bfe5-dcd1b5cdf0b3", "c189bbe9-5a4f-4db1-bad3-a7a05cedb97d",
                       "c82aef5e-1892-433d-9f0c-7d9f0f74de67"],
     "schema_version": "dream.compaction.request.v1", "origin_chain_id": "6a9ef1b2-0dcb-4b22-817f-24d4a03673af"},
    {"theme": "open-loop-4409448b6975", "reason": "reverie_chain_settled", "op_hint": "consolidate",
     "request_id": "compaction-request:51be402f-402d-4891-9989-44fbe73eb224",
     "evidence_refs": ["9a411a3b-3eb6-488e-a92b-5cff3d2f0a08", "dff44679-0c55-4b64-8adc-4ea07401de3f",
                       "40fd635b-07b9-4271-9b01-231793ed6284"],
     "schema_version": "dream.compaction.request.v1", "origin_chain_id": "51be402f-402d-4891-9989-44fbe73eb224"},
    {"theme": "open-loop-e91fa7e1c61e", "reason": "reverie_chain_settled", "op_hint": "consolidate",
     "request_id": "compaction-request:2d12ed33-241b-4394-9ff8-1f8405006103",
     "evidence_refs": ["119806d2-f18f-4c2e-bd46-ff2b4e0db92d", "2c2ae105-cbba-4f7e-a81c-f8c19f3c3fc7"],
     "schema_version": "dream.compaction.request.v1", "origin_chain_id": "2d12ed33-241b-4394-9ff8-1f8405006103"},
    {"theme": "open-loop-e91fa7e1c61e", "reason": "reverie_chain_settled", "op_hint": "consolidate",
     "request_id": "compaction-request:8e8cff28-7d8a-47cc-a14a-1d16a81d79cb",
     "evidence_refs": ["6e4f27f9-0f61-4d40-a181-e459615890e1", "5fe5d23f-1ff8-457a-ab07-46c86b3d6c63"],
     "schema_version": "dream.compaction.request.v1", "origin_chain_id": "8e8cff28-7d8a-47cc-a14a-1d16a81d79cb"},
]
LIVE_SALIENCE = {
    "5d39c104-bdf5-4ec7-bfe5-dcd1b5cdf0b3": 0.601, "c189bbe9-5a4f-4db1-bad3-a7a05cedb97d": 0.601,
    "c82aef5e-1892-433d-9f0c-7d9f0f74de67": 0.601, "9a411a3b-3eb6-488e-a92b-5cff3d2f0a08": 0.742,
    "dff44679-0c55-4b64-8adc-4ea07401de3f": 0.742, "40fd635b-07b9-4271-9b01-231793ed6284": 0.742,
    "119806d2-f18f-4c2e-bd46-ff2b4e0db92d": 0.727, "2c2ae105-cbba-4f7e-a81c-f8c19f3c3fc7": 0.727,
    "6e4f27f9-0f61-4d40-a181-e459615890e1": 0.75, "5fe5d23f-1ff8-457a-ab07-46c86b3d6c63": 0.75,
}
# An unrelated thought nobody asked to compact: it must never move.
BYSTANDER = "00000000-feed-4000-8000-000000000000"


@pytest.fixture
def db(tmp_path, monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    eng = create_engine(f"sqlite:///{tmp_path / 'memory.db'}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE substrate_reverie_thought (thought_id TEXT PRIMARY KEY, "
                       "salience DOUBLE PRECISION, thought_json TEXT)"))
        for tid, sal in {**LIVE_SALIENCE, BYSTANDER: 0.9}.items():
            c.execute(text("INSERT INTO substrate_reverie_thought VALUES (:i, :s, :j)"),
                      {"i": tid, "s": sal, "j": json.dumps({"salience": sal, "producer": "reverie_narrate_v1"})})
    return eng


def salience_of(eng) -> dict[str, float]:
    with eng.connect() as c:
        return {r[0]: r[1] for r in c.execute(text("SELECT thought_id, salience FROM substrate_reverie_thought"))}


def loader_for(eng):
    """What rem_store.load_thought_salience returns (column, original from thought_json), read from ``eng``."""
    def load(ids):
        with eng.connect() as c:
            rows = c.execute(text("SELECT thought_id, salience, thought_json FROM substrate_reverie_thought")).all()
        return {r[0]: (r[1], json.loads(r[2]).get("salience")) for r in rows if r[0] in set(ids)}
    return load


def stage(eng, factor=0.8):
    from app.rem_compaction import build_compaction_delta, consolidate_evidence_ids

    sal = loader_for(eng)(consolidate_evidence_ids(LIVE_REQUESTS))
    return build_compaction_delta(LIVE_REQUESTS, dream_id="dc-test", salience=sal, downscale_factor=factor)


def store_for(eng):
    from app.compaction_store import ReverieSalienceStore

    return ReverieSalienceStore(eng)


# --- the lifecycle ---------------------------------------------------------------------------------------

def test_lifecycle_ask_allow_snapshot_downscale_restore(db):
    from app.compaction_applier import apply_compaction_delta
    from app.compaction_store import restore_snapshot_file

    before = salience_of(db)
    delta = stage(db)
    # REM proposes one downscale per settled thought, x0.8, and no prune.
    assert len(delta.downscale) == 10 and delta.prune == [] and delta.metrics.edges_downscaled == 10
    assert {e.target_id for e in delta.downscale} == {f"substrate_reverie_thought:{t}" for t in LIVE_SALIENCE}
    assert all(e.new_w == round(e.old_w * 0.8, 6) for e in delta.downscale)

    calls: list = []
    r = apply_compaction_delta(delta, consent_for(SHIPPED, calls=calls), store=store_for(db), enabled=True)
    # ask -> allow by the shipped standing consent
    assert calls[0].verb == "compact" and calls[0].consent_request.reversibility == "snapshot_restorable"
    assert (r.consent_verdict, r.consent_rule) == ("allow", "dream.compaction.downscale_only")
    # snapshot -> downscale
    assert r.status == "applied" and r.downscaled == 10 and r.pruned == 0
    assert r.cards_written == 0 and r.cards_skipped_no_gist_store == 4
    snap = json.loads(Path(r.snapshot_path).read_text())
    assert snap["rows"] == LIVE_SALIENCE and snap["delta_id"] == delta.delta_id
    after = salience_of(db)
    assert after[BYSTANDER] == 0.9
    for tid, sal in LIVE_SALIENCE.items():
        assert after[tid] == pytest.approx(sal * 0.8) and after[tid] < sal
    # Relative order among the downscaled is kept (proportional).
    assert sorted(LIVE_SALIENCE, key=LIVE_SALIENCE.get) == sorted(LIVE_SALIENCE, key=after.get)
    # The outcome is on disk next to the snapshot, not only in the logs.
    receipt = json.loads((Path(r.snapshot_path).parent / "receipt.json").read_text())
    assert receipt["status"] == "applied" and receipt["downscaled"] == 10
    # restore -> every value back, from the file the applier wrote
    assert restore_snapshot_file(r.snapshot_path, store_for(db)) == 10
    assert salience_of(db) == before
    # A second restore finds nothing at the planned values: it changes nothing.
    assert restore_snapshot_file(r.snapshot_path, store_for(db)) == 0 and salience_of(db) == before


def test_a_thought_already_downscaled_is_not_proposed_again(db):
    from app.compaction_applier import apply_compaction_delta

    r = apply_compaction_delta(stage(db), consent_for(SHIPPED), store=store_for(db), enabled=True)
    assert r.status == "applied"
    again = stage(db)
    assert again.downscale == [] and len(again.consolidate) == 4
    # A gist-only delta to a store without gists is empty: no snapshot, nothing written.
    lowered = salience_of(db)
    r2 = apply_compaction_delta(again, consent_for(SHIPPED), store=store_for(db), enabled=True)
    assert r2.status == "empty" and r2.snapshot_path is None and r2.cards_skipped_no_gist_store == 4
    assert salience_of(db) == lowered


def test_a_failed_apply_never_undoes_another_writer(db):
    """Review finding 1: A snapshots, then B lowers one of A's rows, then A fails on that row. A's rollback
    must leave B's value, and A must have written nothing (one transaction)."""
    from app.compaction_applier import apply_compaction_delta

    before = salience_of(db)
    a_delta = stage(db)
    shared = a_delta.downscale[3]
    tid = shared.target_id.split(":", 1)[1]
    store = store_for(db)
    real_snapshot = store.snapshot

    def snapshot_then_b_writes(delta):
        snap = real_snapshot(delta)
        store_for(db).downscale(shared.target_id, shared.old_w, 0.1)  # writer B, after A's snapshot
        return snap

    store.snapshot = snapshot_then_b_writes
    r = apply_compaction_delta(a_delta, consent_for(SHIPPED), store=store, enabled=True)
    assert r.status == "rolled_back" and r.downscaled == 0
    assert salience_of(db) == {**before, tid: 0.1}


def test_a_stale_target_writes_nothing_even_when_restore_cannot_run(db):
    """Review finding 2: the batch is one transaction. With restore broken, a stale row still leaves every
    other row untouched (a killed container mid-apply leaves no half delta), and the status says so."""
    from app.compaction_applier import apply_compaction_delta

    before = salience_of(db)
    delta = stage(db)
    stale = delta.downscale[-1].target_id.split(":", 1)[1]
    with db.begin() as c:
        c.execute(text("UPDATE substrate_reverie_thought SET salience = 0.5 WHERE thought_id = :i"), {"i": stale})
    store = store_for(db)

    def broken_restore(snapshot):
        raise RuntimeError("restore down")

    store.restore = broken_restore
    r = apply_compaction_delta(delta, consent_for(SHIPPED), store=store, enabled=True)
    assert r.status == "rollback_failed" and r.downscaled == 0
    assert salience_of(db) == {**before, stale: 0.5}


def test_a_stale_target_rolls_the_whole_apply_back(db):
    from app.compaction_applier import apply_compaction_delta

    before = salience_of(db)
    delta = stage(db)
    stale = delta.downscale[-1].target_id.split(":", 1)[1]
    with db.begin() as c:  # something else moved one thought between staging and apply
        c.execute(text("UPDATE substrate_reverie_thought SET salience = 0.5 WHERE thought_id = :i"), {"i": stale})
    r = apply_compaction_delta(delta, consent_for(SHIPPED), store=store_for(db), enabled=True)
    assert r.status == "rolled_back" and "stale target" in r.error
    assert salience_of(db) == {**before, stale: 0.5}


def test_a_missing_target_fails_the_snapshot_and_writes_nothing(db):
    from app.compaction_applier import apply_compaction_delta

    delta = stage(db)
    gone = delta.downscale[0].target_id.split(":", 1)[1]
    with db.begin() as c:
        c.execute(text("DELETE FROM substrate_reverie_thought WHERE thought_id = :i"), {"i": gone})
    before = salience_of(db)
    r = apply_compaction_delta(delta, consent_for(SHIPPED), store=store_for(db), enabled=True)
    assert r.status == "rolled_back" and r.error.startswith("snapshot_failed") and r.downscaled == 0
    assert salience_of(db) == before


# --- the guards ------------------------------------------------------------------------------------------

def test_prune_is_refused_with_the_standing_consent_present(db):
    """Regression: the shipped standing consent allows downscale only. A plan that prunes is refused by the
    market, and the store refuses prune on its own too."""
    from app.compaction_applier import apply_compaction_delta
    from app.compaction_store import StoreRefused
    from orion.schemas.compaction import PruneEntryV1

    assert "dream-compaction-downscale" in SHIPPED.standing
    before = salience_of(db)
    delta = stage(db).model_copy(update={"prune": [PruneEntryV1(episodic_id=BYSTANDER, salience=0.02)]})
    calls: list = []
    r = apply_compaction_delta(delta, consent_for(SHIPPED, calls=calls), store=store_for(db), enabled=True,
                               downscale_only=False)
    assert calls[0].verb == "prune" and calls[0].consent_request.reversibility == "irreversible"
    assert r.status == "not_approved" and r.consent_verdict == "refuse" and r.consent_rule == "dream.compaction.prune"
    assert salience_of(db) == before
    with pytest.raises(StoreRefused):
        store_for(db).prune(BYSTANDER)


def test_no_standing_consent_writes_nothing(db):
    from app.compaction_applier import apply_compaction_delta

    before = salience_of(db)
    r = apply_compaction_delta(stage(db), consent_for(NO_STANDING), store=store_for(db), enabled=True)
    assert r.status == "not_approved" and r.consent_verdict == "needs_consent" and r.snapshot_path is None
    assert salience_of(db) == before


def test_apply_off_writes_nothing_even_with_consent(db):
    from app.compaction_applier import apply_compaction_delta

    before = salience_of(db)
    r = apply_compaction_delta(stage(db), consent_for(GRANTED), store=store_for(db), enabled=False)
    assert r.status == "disabled" and salience_of(db) == before


@pytest.mark.parametrize("target,old,new,why", [
    ("substrate_reverie_thought:9a411a3b-3eb6-488e-a92b-5cff3d2f0a08", 0.742, 0.9, "must lower"),
    ("substrate_reverie_thought:9a411a3b-3eb6-488e-a92b-5cff3d2f0a08", 0.742, 0.742, "must lower"),
    ("substrate_reverie_thought:9a411a3b-3eb6-488e-a92b-5cff3d2f0a08", 0.742, -0.1, "must lower"),
    ("memory_cards:9a411a3b-3eb6-488e-a92b-5cff3d2f0a08", 0.742, 0.5, "not a reverie-thought target"),
    ("substrate_reverie_thought:", 0.742, 0.5, "empty reverie-thought id"),
])
def test_the_store_refuses_anything_but_a_reverie_downscale(db, target, old, new, why):
    from app.compaction_store import StoreRefused

    before = salience_of(db)
    from orion.schemas.compaction import DownscaleEntryV1

    with pytest.raises(StoreRefused, match=why):
        store_for(db).downscale(target, old, new)
    ok = DownscaleEntryV1(target_id="substrate_reverie_thought:dff44679-0c55-4b64-8adc-4ea07401de3f",
                          old_w=0.742, new_w=0.5)
    bad = DownscaleEntryV1.model_construct(target_id=target, old_w=old, new_w=new, reason="")
    with pytest.raises(StoreRefused, match=why):  # one bad entry refuses the whole batch
        store_for(db).downscale_many([ok, bad])
    with pytest.raises(StoreRefused):
        store_for(db).write_gist("card", [], [])
    assert salience_of(db) == before


def test_restore_refuses_a_foreign_or_unplanned_snapshot(db):
    from app.compaction_store import SNAPSHOT_SCHEMA, StoreRefused

    with pytest.raises(StoreRefused):
        store_for(db).restore({"schema": "other", "table": "memory_cards", "column": "salience", "rows": {}})
    with pytest.raises(StoreRefused, match="planned"):
        store_for(db).restore({"schema": SNAPSHOT_SCHEMA, "table": "substrate_reverie_thought",
                               "column": "salience", "rows": {BYSTANDER: 0.9}})


def test_downscale_entries_skip_unknown_lowered_and_bad_factor():
    from app.rem_compaction import downscale_entries

    themed = [("t", ["a", "b", "c", "d", "e", "a"])]
    sal = {"a": (0.6, 0.6), "b": (0.48, 0.6), "c": (0.7, None), "d": (0.0, 0.0), "e": (1e-7, 1e-7)}
    out = downscale_entries(themed, sal, 0.5)
    assert [(e.target_id, e.old_w, e.new_w) for e in out] == [("substrate_reverie_thought:a", 0.6, 0.3)]
    assert downscale_entries(themed, sal, 1.0) == [] and downscale_entries(themed, sal, 0.0) == []


def test_a_salience_load_failure_still_stages_the_gist_delta(monkeypatch):
    import asyncio

    from app import rem_compaction as rc
    from app.settings import settings

    monkeypatch.setattr(settings, "ORION_DREAM_REM_ENABLED", True)

    def boom(ids):
        raise RuntimeError("db down")

    d = asyncio.run(rc.run_rem_compaction_once(None, request_loader=lambda n: LIVE_REQUESTS,
                                               delta_persister=lambda d: True, salience_loader=boom))
    assert d is not None and len(d.consolidate) == 4 and d.downscale == []


# --- shipped defaults and wiring ---------------------------------------------------------------------------

def test_apply_ships_on_with_a_persistent_snapshot_dir():
    """Shipped code defaults (not this host's env)."""
    from pydantic import ValidationError

    from app.settings import Settings

    f = Settings.model_fields
    assert f["ORION_DREAM_COMPACTION_APPLY_ENABLED"].default is True
    assert f["ORION_DREAM_COMPACTION_DOWNSCALE_ONLY"].default is True
    assert f["DREAM_COMPACTION_SNAPSHOT_DIR"].default.startswith("/app/logs/")
    assert f["DREAM_COMPACTION_DOWNSCALE_FACTOR"].default == 0.8
    for bad in (0.0, 1.0, 8.0):
        with pytest.raises(ValidationError):
            Settings(DREAM_COMPACTION_DOWNSCALE_FACTOR=bad)


def test_the_sleep_cycle_loads_salience_stages_persists_then_applies(monkeypatch):
    """Behavior of app.main's real ``_rem``: salience is read for the requests' thoughts, the delta carries
    downscale entries, and the apply comes after the persist, only for a stored delta."""
    import asyncio

    from app import compaction_consent, main, rem_store
    from app.settings import settings

    monkeypatch.setattr(settings, "ORION_DREAM_REM_ENABLED", True)
    events: list = []

    async def no_bus():
        return None

    async def apply(delta):
        events.append(("apply", delta.delta_id))

    monkeypatch.setattr(main, "_cycle_bus", no_bus)
    monkeypatch.setattr(rem_store, "load_pending_requests", lambda n, since=None: LIVE_REQUESTS)
    monkeypatch.setattr(rem_store, "load_thought_salience",
                        lambda ids: events.append(("salience", len(ids))) or {t: (v, v) for t, v in LIVE_SALIENCE.items()})
    monkeypatch.setattr(rem_store, "persist_compaction_delta",
                        lambda d: events.append(("persist", d.delta_id, len(d.downscale))) or True)
    monkeypatch.setattr(compaction_consent, "ask_consent_and_apply", apply)
    delta_id = asyncio.run(main.build_cycle_deps().rem_compaction("dc-test", None))
    assert events == [("salience", 10), ("persist", delta_id, 10), ("apply", delta_id)]
    events.clear()
    monkeypatch.setattr(rem_store, "persist_compaction_delta", lambda d: False)
    asyncio.run(main.build_cycle_deps().rem_compaction("dc-test", None))
    assert [e[0] for e in events] == ["salience"]  # not stored -> not applied


def test_the_preview_endpoint_asks_with_no_store(monkeypatch):
    import asyncio

    from app import compaction_consent, main, rem_store
    from app.settings import settings

    monkeypatch.setattr(settings, "ORION_DREAM_REM_ENABLED", True)
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", False)
    seen: list = []

    def spy(delta, consent=None, *, store, **kw):
        seen.append(store)
        from app.compaction_applier import CompactionApplyReceiptV1

        return CompactionApplyReceiptV1(delta_id=delta.delta_id, status="no_store")

    monkeypatch.setattr(rem_store, "load_pending_requests", lambda n, since=None: LIVE_REQUESTS)
    monkeypatch.setattr(rem_store, "load_thought_salience", lambda ids: {t: (v, v) for t, v in LIVE_SALIENCE.items()})
    monkeypatch.setattr(rem_store, "persist_compaction_delta", lambda d: True)
    monkeypatch.setattr(compaction_consent, "apply_compaction_delta", spy)
    out = asyncio.run(main.rem_preview_endpoint())
    assert out["status"] == "staged" and out["applied"] is False and seen == [None]
