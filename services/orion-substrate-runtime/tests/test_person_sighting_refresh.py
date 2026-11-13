"""A sighting refreshes the named camera cluster node (2026-11-03).

docs/superpowers/specs/2026-11-03-person-graph-sighting-refresh-design.md acceptance
1, 2, 4, 6 and the kill switch. Real adapter, real refresh step, real
SubstrateDynamicsEngine over an in-memory store; the Falkor codec round trip covers
the durable ``label_set_at`` property.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import (
    EntityNodeV1,
    SubstrateActivationV1,
    SubstrateGraphRecordV1,
)
from orion.substrate.adapters.vision_individual import (
    LABEL_SET_AT_KEY,
    SIGHTING_HALF_LIFE_SECONDS,
    entity_node_id_for_individual,
    label_set_at,
    map_vision_individual_label_to_substrate,
    map_walkway_day_seen_to_substrate,
    sighting_refreshed_entity,
)
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.store import InMemorySubstrateGraphStore

NAMED = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
SEEN = datetime(2026, 11, 2, 17, 22, tzinfo=timezone.utc)
SAM = "person-0000aaaa-0000-5000-8000-000000000001"
ROBIN = "person-0000aaaa-0000-5000-8000-000000000002"


def _cluster(iid: str = "c1", *, label: str = "Sam", answered_at: datetime = NAMED) -> EntityNodeV1:
    return map_vision_individual_label_to_substrate(
        individual_id=iid, label=label, kind="person", ask_id=f"ask-{iid}", answered_at=answered_at).nodes[0]


def _legacy_cluster(iid: str = "c1") -> EntityNodeV1:
    """A node as live had it before this patch: no half-life, no label_set_at."""
    node = _cluster(iid)
    metadata = {k: v for k, v in node.metadata.items() if k != LABEL_SET_AT_KEY}
    return node.model_copy(update={
        "metadata": metadata,
        "signals": node.signals.model_copy(update={"activation": SubstrateActivationV1()}),
    })


# --- acceptance 1: the pure refresh ---------------------------------------------------------

def test_refresh_moves_observed_at_forward_and_sets_half_life_and_label_stamp():
    legacy = _legacy_cluster()
    out = sighting_refreshed_entity(legacy, SEEN)
    assert out is not None
    assert out.temporal.observed_at == SEEN
    assert out.signals.activation.decay_half_life_seconds == SIGHTING_HALF_LIFE_SECONDS
    assert out.metadata[LABEL_SET_AT_KEY] == NAMED.isoformat()   # the naming time, not the sighting
    assert out.signals.activation.activation == legacy.signals.activation.activation


def test_refresh_never_moves_observed_at_backwards_and_is_none_when_nothing_changes():
    node = sighting_refreshed_entity(_legacy_cluster(), SEEN)
    assert sighting_refreshed_entity(node, SEEN) is None
    assert sighting_refreshed_entity(node, SEEN - timedelta(days=3)) is None
    assert sighting_refreshed_entity(node, None) is None
    later = sighting_refreshed_entity(node, SEEN + timedelta(hours=2))
    assert later.temporal.observed_at == SEEN + timedelta(hours=2)
    assert later.metadata[LABEL_SET_AT_KEY] == NAMED.isoformat()   # backfilled once, kept


def test_a_naive_last_seen_at_is_read_as_utc():
    out = sighting_refreshed_entity(_legacy_cluster(), SEEN.replace(tzinfo=None))
    assert out.temporal.observed_at == SEEN


def test_new_clusters_and_new_day_nodes_carry_the_sighting_half_life():
    assert _cluster().signals.activation.decay_half_life_seconds == SIGHTING_HALF_LIFE_SECONDS
    assert label_set_at(_cluster()) == NAMED
    day = map_walkway_day_seen_to_substrate(
        stream_id="walkway", day=SEEN.date(), line="Sam at 17:22",
        seen=[(entity_node_id_for_individual("c1"), "Sam", "17:22")], observed_at=SEEN).nodes[0]
    assert day.signals.activation.decay_half_life_seconds == SIGHTING_HALF_LIFE_SECONDS


def test_label_set_at_survives_the_falkor_codec_round_trip():
    from orion.substrate.falkor_codec import decode_node, encode_node_properties

    node = sighting_refreshed_entity(_legacy_cluster(), SEEN)
    props = encode_node_properties(node, "identity")
    assert props[LABEL_SET_AT_KEY] == NAMED.isoformat()
    back = decode_node(props)
    assert label_set_at(back) == NAMED and back.temporal.observed_at == SEEN
    assert back.signals.activation.decay_half_life_seconds == SIGHTING_HALF_LIFE_SECONDS
    # A writer that does not carry the stamp leaves it out of the SET clause (never clears it).
    assert LABEL_SET_AT_KEY not in encode_node_properties(_legacy_cluster(), "identity")


def test_label_set_at_is_a_returned_falkor_field():
    from orion.substrate.falkor_store import NATIVE_NODE_RETURN_FIELDS

    assert LABEL_SET_AT_KEY in NATIVE_NODE_RETURN_FIELDS


# --- acceptance 2: lifecycle on the real dynamics engine ------------------------------------

def _store_with(node: EntityNodeV1) -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    SubstrateGraphMaterializer(store=store).apply_record(
        SubstrateGraphRecordV1(anchor_scope="world", subject_ref=node.subject_ref, nodes=[node]))
    return store


def _refresh(store, last_seen_at, *, iid: str = "c1", person: str = SAM) -> int:
    from app.person_graph_loop import LinkRow, PersonRow, _refresh_sightings

    refreshed, _ = asyncio.run(_refresh_sightings(
        [PersonRow(person, "Sam")], [LinkRow("l1", person, iid, last_seen_at=last_seen_at)], store=store))
    return refreshed


def _activation(store, iid: str = "c1") -> float:
    return store.get_node_by_id(entity_node_id_for_individual(iid)).signals.activation.activation


def _run(store, engine, *, start: datetime, until: datetime, step=timedelta(minutes=10), last_seen=None):
    t = start
    while t <= until:
        if last_seen is not None:
            _refresh(store, last_seen if last_seen <= t else None)
        engine.tick(now=t)
        t += step


def test_a_sighting_lifts_the_cluster_and_it_returns_to_rest_within_a_day():
    store = _store_with(_legacy_cluster())
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=NAMED + timedelta(minutes=1))   # birth tick: the frozen 0.625 live has today
    assert abs(_activation(store) - 0.625) < 0.02
    _run(store, engine, start=NAMED + timedelta(minutes=10), until=SEEN - timedelta(minutes=10),
         step=timedelta(hours=6), last_seen=NAMED)   # ~13 days with no sighting: decays to rest
    assert _activation(store) <= 0.08
    readings = {}
    t = SEEN + timedelta(minutes=10)
    while t <= SEEN + timedelta(hours=20):
        _refresh(store, SEEN)
        engine.tick(now=t)
        readings[t - SEEN] = _activation(store)
        t += timedelta(minutes=10)
    assert readings[timedelta(minutes=10)] >= 0.5
    assert readings[timedelta(hours=6)] < 0.35
    assert readings[timedelta(hours=20)] <= 0.08
    # "Rest" is the activation reading above. The dormant FLAG is not set on
    # entity nodes: the durable store keeps it on concept nodes only
    # (store.DORMANCY_NODE_KINDS), so the tick no longer decides it here.
    node = store.get_node_by_id(entity_node_id_for_individual("c1"))
    assert "dormant" not in node.metadata


def test_the_old_frozen_shape_fails_the_lifecycle_check():
    """Today's shape (no half-life, observed_at stuck at naming) never reaches rest."""
    store = _store_with(_legacy_cluster())
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=NAMED + timedelta(minutes=1))
    _run(store, engine, start=SEEN + timedelta(minutes=10), until=SEEN + timedelta(hours=20),
         step=timedelta(hours=1))
    assert _activation(store) > 0.6   # the ratchet: what the lifecycle test above rejects


def test_a_never_seen_cluster_reaches_rest():
    store = _store_with(_cluster())
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=NAMED + timedelta(minutes=1))
    _run(store, engine, start=NAMED + timedelta(hours=1), until=NAMED + timedelta(hours=24),
         step=timedelta(hours=1), last_seen=None)
    assert _activation(store) <= 0.08


# --- acceptance 4: flood guard / idempotence ------------------------------------------------

def test_two_passes_with_no_new_sighting_change_nothing():
    store = _store_with(_legacy_cluster())
    assert _refresh(store, SEEN) == 1
    before = store.get_node_by_id(entity_node_id_for_individual("c1")).model_dump()
    assert _refresh(store, SEEN) == 0
    assert _refresh(store, SEEN - timedelta(hours=1)) == 0
    assert store.get_node_by_id(entity_node_id_for_individual("c1")).model_dump() == before


def test_the_refresh_never_writes_activation():
    store = _store_with(_legacy_cluster())
    SubstrateDynamicsEngine(store=store).tick(now=NAMED + timedelta(minutes=1))
    before = store.get_node_by_id(entity_node_id_for_individual("c1")).signals.activation
    _refresh(store, SEEN)
    after = store.get_node_by_id(entity_node_id_for_individual("c1")).signals.activation
    assert (after.activation, after.recency_score) == (before.activation, before.recency_score)


def test_copied_unlinked_and_not_a_person_clusters_are_not_refreshed():
    from app.person_graph_loop import LinkRow, PersonRow, _refresh_sightings

    store = _store_with(_legacy_cluster("c1"))
    for iid in ("c2", "c3"):
        SubstrateGraphMaterializer(store=store).apply_record(SubstrateGraphRecordV1(
            anchor_scope="world", subject_ref=f"vision_individual:{iid}", nodes=[_legacy_cluster(iid)]))
    persons = [PersonRow(SAM, "Sam"), PersonRow(ROBIN, "Robin", status="not_a_person")]
    links = [LinkRow("l1", SAM, "c1", label_copied=True, last_seen_at=SEEN),
             LinkRow("l2", ROBIN, "c2", last_seen_at=SEEN),
             LinkRow("l3", "person-gone", "c3", last_seen_at=SEEN)]
    assert asyncio.run(_refresh_sightings(persons, links, store=store)) == (0, 0)


def test_named_clusters_outside_the_live_links_get_the_half_life_only():
    """Review 2026-11-03: belonging-linked, not_a_person and unlinked named clusters kept the
    frozen 0.625 forever (6 live). They rest now; their observed_at does not move."""
    from app.person_graph_loop import LinkRow, PersonRow, _refresh_sightings

    store = _store_with(_legacy_cluster("c1"))
    for iid in ("c2", "c3"):
        SubstrateGraphMaterializer(store=store).apply_record(SubstrateGraphRecordV1(
            anchor_scope="world", subject_ref=f"vision_individual:{iid}", nodes=[_legacy_cluster(iid)]))
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=NAMED + timedelta(minutes=1))
    links = [LinkRow("l1", SAM, "c1", last_seen_at=SEEN)]
    out = asyncio.run(_refresh_sightings([PersonRow(SAM, "Sam")], links, store=store,
                                         named_cluster_ids=("c1", "c2", "c3", "c-no-node")))
    assert out == (1, 2)
    for iid in ("c2", "c3"):
        node = store.get_node_by_id(entity_node_id_for_individual(iid))
        assert node.temporal.observed_at == NAMED
        assert node.signals.activation.decay_half_life_seconds == SIGHTING_HALF_LIFE_SECONDS
    assert asyncio.run(_refresh_sightings([PersonRow(SAM, "Sam")], links, store=store,
                                          named_cluster_ids=("c1", "c2", "c3"))) == (0, 0)
    engine.tick(now=SEEN)
    assert _activation(store, "c2") <= 0.08 and _activation(store, "c3") <= 0.08


def test_the_refresh_names_activation_state_as_not_owned():
    """The refresh writes back the activation it just read, so comparing values cannot tell a
    skipped write from an unskipped one. Check the contract on the write itself."""
    from orion.substrate.falkor_codec import ACTIVATION_STATE_KEYS

    store = _store_with(_legacy_cluster())
    calls = []
    real = store.upsert_node

    def recording(**kw):
        calls.append(kw.get("skip_metadata_keys") or frozenset())
        return real(**kw)

    store.upsert_node = recording
    assert _refresh(store, SEEN) == 1
    assert calls and all(ACTIVATION_STATE_KEYS <= keys for keys in calls)


def test_a_rename_is_not_a_sighting():
    """Review 2026-11-03: a rename used to move observed_at to the answer time, which now
    reads as "seen" and would wake every cluster of a bulk rename."""
    from orion.substrate.adapters.vision_individual import relabelled_individual_entity

    store = _store_with(_legacy_cluster())
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=NAMED + timedelta(minutes=1))
    engine.tick(now=SEEN)   # still the frozen value: no half-life yet
    existing = store.get_node_by_id(entity_node_id_for_individual("c1"))
    renamed = relabelled_individual_entity(existing, _cluster(label="Robin", answered_at=SEEN))
    assert renamed.temporal.observed_at == existing.temporal.observed_at
    assert label_set_at(renamed) == SEEN
    from orion.substrate.falkor_codec import ACTIVATION_STATE_KEYS, EXTERNALLY_OWNED_METADATA_KEYS
    store.upsert_node(identity_key=None, node=renamed,
                      skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS | ACTIVATION_STATE_KEYS)
    engine.tick(now=SEEN + timedelta(minutes=10))
    assert _activation(store) <= 0.08


def _day_node_world():
    store = _store_with(_legacy_cluster())
    day = map_walkway_day_seen_to_substrate(
        stream_id="walkway", day=SEEN.date(), line="Sam at 17:22",
        seen=[(entity_node_id_for_individual("c1"), "Sam", "17:22")],
        observed_at=SEEN.replace(hour=6, minute=0))
    SubstrateGraphMaterializer(store=store).apply_record(day)
    _refresh(store, SEEN)
    return store, day.nodes[0].node_id


def test_day_node_echoes_the_sighting_then_returns_to_rest():
    """Pins the day node's response (2026-11-03, propagation decay mode ``source``).

    The cluster's activation reaches the day node through its observed_in edge
    (confidence 0.8) at the default attenuation 0.6 and is NOT decayed again by the
    day node's own age (its observed_at is the day's start, 11 h earlier). Before this
    fix it read ~0.04. It still returns to rest once the cluster goes quiet."""
    store, day_id = _day_node_world()
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=SEEN + timedelta(minutes=5))
    cluster = _activation(store)
    day_value = store.get_node_by_id(day_id).signals.activation.activation
    assert cluster >= 0.5
    assert abs(day_value - cluster * 0.6 * 0.8) < 0.01
    assert day_value > 0.2
    engine.tick(now=SEEN + timedelta(hours=20))
    assert store.get_node_by_id(day_id).signals.activation.activation <= 0.08


def test_day_node_rollback_mode_keeps_the_old_muted_echo():
    """SUBSTRATE_DYNAMICS_PROPAGATION_DECAY_MODE=receiver_age: the pre-fix double decay."""
    store, day_id = _day_node_world()
    engine = SubstrateDynamicsEngine(store=store, propagation_decay_mode="receiver_age")
    engine.tick(now=SEEN + timedelta(minutes=5))
    assert _activation(store) >= 0.5
    assert store.get_node_by_id(day_id).signals.activation.activation < 0.1   # ~0.075, muted by its 11 h age
    engine.tick(now=SEEN + timedelta(hours=20))
    assert store.get_node_by_id(day_id).signals.activation.activation <= 0.08


# --- acceptance 6 + kill switch: through the full pass --------------------------------------

def _world():
    from app.person_graph_loop import PRODUCER
    from orion.substrate.assertion_projector import AssertionProjector
    from orion.substrate.reader_capability import ALWAYS_READY
    from test_person_graph_loop import FakeJournal

    store = _store_with(_legacy_cluster())
    journal = FakeJournal()
    projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                   readiness=ALWAYS_READY, proposal_actors=(PRODUCER,))
    return store, journal, projector


def _sync(store, journal, projector, links, *, person=SAM, name="Sam", now=SEEN + timedelta(minutes=5), **kw):
    from app.person_graph_loop import PersonRow, sync_person_graph

    return asyncio.run(sync_person_graph(persons=[PersonRow(person, name)], links=links, journal=journal,
                                         store=store, projector=projector, now=now, **kw))


def test_a_cluster_refreshed_by_a_sighting_can_still_be_renamed():
    from app.person_graph_loop import LinkRow

    store, journal, projector = _world()
    seen = [LinkRow("l1", SAM, "c1", label="Sam", kind="person", labeled_at=NAMED, last_seen_at=SEEN)]
    assert _sync(store, journal, projector, seen).sightings_refreshed == 1
    node = store.get_node_by_id(entity_node_id_for_individual("c1"))
    assert node.temporal.observed_at == SEEN and label_set_at(node) == NAMED
    # Renamed in Postgres a day after naming but BEFORE the sighting: the old observed_at
    # guard (observed_at > labeled_at) would have refused this rename forever.
    renamed = [LinkRow("l2", ROBIN, "c1", label="Robin", kind="person", label_ask_id="ask-new",
                       labeled_at=NAMED + timedelta(days=1), last_seen_at=SEEN)]
    report = _sync(store, journal, projector, renamed, person=ROBIN, name="Robin")
    node = store.get_node_by_id(entity_node_id_for_individual("c1"))
    assert report.clusters_relabelled == 1
    assert node.label == "Robin" and "Sam" in node.aliases
    assert node.temporal.observed_at == SEEN   # the rename does not undo the sighting
    assert node.signals.activation.decay_half_life_seconds == SIGHTING_HALF_LIFE_SECONDS


def test_a_bus_rename_postgres_has_not_applied_yet_is_still_kept():
    from app.person_graph_loop import LinkRow
    from orion.substrate.adapters.vision_individual import relabelled_individual_entity

    store, journal, projector = _world()
    existing = store.get_node_by_id(entity_node_id_for_individual("c1"))
    bus_rename = relabelled_individual_entity(existing, _cluster(label="Robin", answered_at=SEEN))
    store.upsert_node(identity_key=None, node=bus_rename)
    stale = [LinkRow("l1", SAM, "c1", label="Sam", kind="person", labeled_at=NAMED + timedelta(days=1),
                     last_seen_at=SEEN)]
    report = _sync(store, journal, projector, stale)
    assert report.clusters_relabelled == 0
    assert store.get_node_by_id(entity_node_id_for_individual("c1")).label == "Robin"


def test_flag_off_writes_no_sighting():
    from app.person_graph_loop import LinkRow

    store, journal, projector = _world()
    before = store.get_node_by_id(entity_node_id_for_individual("c1")).model_dump()
    report = _sync(store, journal, projector, [LinkRow("l1", SAM, "c1", last_seen_at=SEEN)],
                   refresh_sightings=False)
    assert report.sightings_refreshed == 0
    after = store.get_node_by_id(entity_node_id_for_individual("c1")).model_dump()
    assert after["temporal"] == before["temporal"] and after["signals"] == before["signals"]


def test_settings_ship_the_flag_on(monkeypatch):
    from app.settings import Settings

    monkeypatch.setenv("POSTGRES_URI", "postgresql://x/y")
    monkeypatch.delenv("PERSON_SIGHTING_REFRESH_ENABLED", raising=False)
    assert Settings().person_sighting_refresh_enabled is True
    monkeypatch.setenv("PERSON_SIGHTING_REFRESH_ENABLED", "false")
    assert Settings().person_sighting_refresh_enabled is False


def test_the_loop_hands_the_flag_to_the_pass(monkeypatch):
    """run_person_graph_loop reads PERSON_SIGHTING_REFRESH_ENABLED and passes it to each pass."""
    import sys
    import types

    from app import person_graph_loop

    class _Pool:
        async def close(self):
            return None

    async def _create_pool(**_kw):
        return _Pool()

    async def _read_rows(_pool):
        return [], []

    for flag in (True, False):
        seen: list[bool] = []
        stop = asyncio.Event()

        async def _fake_sync(*, refresh_sightings, named_cluster_ids, **_kw):
            seen.append(refresh_sightings)
            assert named_cluster_ids == (("c1",) if refresh_sightings else ())
            stop.set()

        monkeypatch.setitem(sys.modules, "asyncpg", types.SimpleNamespace(create_pool=_create_pool))
        monkeypatch.setattr(person_graph_loop, "read_rows", _read_rows)

        async def _read_named(_pool):
            return ("c1",)

        monkeypatch.setattr(person_graph_loop, "read_named_cluster_ids", _read_named)
        monkeypatch.setattr(person_graph_loop, "sync_person_graph", _fake_sync)
        settings = types.SimpleNamespace(person_graph_enabled=True, person_graph_interval_sec=0.01,
                                         person_sighting_refresh_enabled=flag, postgres_uri="postgresql://x/y")
        asyncio.run(person_graph_loop.run_person_graph_loop(get_store=lambda: object(), settings=settings,
                                                            stop_event=stop))
        assert seen == [flag]


# --- real FalkorDB: the durable stamp and the tick-owned activation (CI, must not skip) -----

FALKOR = __import__("os").getenv("ORION_TEST_FALKOR_URI", "")


@__import__("pytest").mark.skipif(not FALKOR, reason="ORION_TEST_FALKOR_URI not set (throwaway FalkorDB only)")
def test_real_falkor_refresh_round_trips_and_the_tick_still_owns_activation():
    """Falkor keeps no free-form entity metadata: only the native label_set_at property makes
    the relabel guard's stamp survive a restart. The refresh must not touch activation."""
    import uuid

    import redis

    from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

    graph = f"orion_test_sighting_refresh_{uuid.uuid4().hex[:10]}"
    cfg = FalkorSubstrateStoreConfig(uri=FALKOR, graph_name=graph)
    nid = entity_node_id_for_individual("c1")
    try:
        store = FalkorSubstrateStore(cfg, hydrate=False)
        legacy = _legacy_cluster()
        SubstrateGraphMaterializer(store=store).apply_record(
            SubstrateGraphRecordV1(anchor_scope="world", subject_ref=legacy.subject_ref, nodes=[legacy]))
        SubstrateDynamicsEngine(store=store).tick(now=NAMED + timedelta(minutes=1))
        born = FalkorSubstrateStore(cfg, hydrate=True).get_node_by_id(nid).signals.activation.activation
        assert abs(born - 0.625) < 0.02
        assert _refresh(store, SEEN) == 1

        fresh = FalkorSubstrateStore(cfg, hydrate=True)
        node = fresh.get_node_by_id(nid)
        assert node.temporal.observed_at == SEEN and label_set_at(node) == NAMED
        assert node.signals.activation.decay_half_life_seconds == SIGHTING_HALF_LIFE_SECONDS
        assert node.signals.activation.activation == born          # the refresh wrote no activation
        assert _refresh(fresh, SEEN) == 0                            # idempotent after a restart

        # A writer that does not carry the stamp never clears it (ON MATCH omits it).
        stampless = node.model_copy(update={"metadata": {k: v for k, v in node.metadata.items()
                                                         if k != LABEL_SET_AT_KEY}})
        fresh.upsert_node(identity_key=None, node=stampless)
        assert label_set_at(FalkorSubstrateStore(cfg, hydrate=True).get_node_by_id(nid)) == NAMED

        engine = SubstrateDynamicsEngine(store=fresh)
        engine.tick(now=SEEN + timedelta(minutes=10))
        assert fresh.get_node_by_id(nid).signals.activation.activation >= 0.5
        engine.tick(now=SEEN + timedelta(hours=20))
        assert FalkorSubstrateStore(cfg, hydrate=True).get_node_by_id(nid).signals.activation.activation <= 0.08
    finally:
        try:
            redis.Redis.from_url(FALKOR).execute_command("GRAPH.DELETE", graph)
        except redis.ResponseError:  # graph never created: keep the real failure visible
            pass
