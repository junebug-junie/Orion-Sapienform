"""L6 (2026-10-06): the dynamics tick must decay stored activation once per unit
of real time, not re-decay an already-decayed value by the node's full age every
tick. See docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md, L6."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.core.activation_decay import decay_activation
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS,
    EvidenceNodeV1,
    SubstrateActivationV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.activation import (
    ACTIVATION_DECAYED_AT_KEY,
    activation_decay_anchor,
    normalize_decay_mode,
)
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.falkor_codec import (
    DYNAMICS_ENGINE_OWNED_METADATA_KEYS,
    decode_node,
    encode_node_properties,
)
from orion.substrate.falkor_store import (
    NATIVE_NODE_RETURN_FIELDS,
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    RecordingFalkorClient,
)
from orion.substrate.store import InMemorySubstrateGraphStore

T0 = datetime(2026, 10, 6, 3, 0, 0, tzinfo=timezone.utc)
HL = DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS  # 30 days, the live default


def _concept(
    *,
    node_id: str = "sub-concept-seed-test",
    activation: float = 1.0,
    observed_at: datetime = T0 - timedelta(hours=23),
    half_life: int | None = HL,
    salience: float = 0.0,
    metadata: dict | None = None,
) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        label=node_id,
        anchor_scope="orion",
        temporal={"observed_at": observed_at},
        signals=SubstrateSignalBundleV1(
            confidence=0.8,
            salience=salience,
            activation=SubstrateActivationV1(
                activation=activation,
                recency_score=0.0,
                decay_half_life_seconds=half_life,
                decay_floor=0.0,
            ),
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="test", source_channel="test", producer="test"
        ),
        metadata=dict(metadata or {}),
    )


class _CountingStore(InMemorySubstrateGraphStore):
    def __init__(self) -> None:
        super().__init__()
        self.upserts = 0

    def upsert_node(self, *, identity_key, node, skip_metadata_keys=None) -> None:
        self.upserts += 1
        super().upsert_node(identity_key=identity_key, node=node, skip_metadata_keys=skip_metadata_keys)


def _store_with(node: ConceptNodeV1) -> _CountingStore:
    store = _CountingStore()
    store.upsert_node(identity_key=f"id:{node.node_id}", node=node)
    store.upserts = 0
    return store


def _activation(store: InMemorySubstrateGraphStore, node_id: str) -> float:
    return store.get_node_by_id(node_id).signals.activation.activation


def _run_ticks(store, *, mode: str, start: datetime, step_s: float, n: int) -> None:
    engine = SubstrateDynamicsEngine(store=store, decay_mode=mode)
    for i in range(1, n + 1):
        engine.tick(now=start + timedelta(seconds=step_s * i))


# --- the bug itself ----------------------------------------------------------


def test_one_tick_equals_n_ticks_over_same_elapsed_time() -> None:
    """20 ticks 30 s apart must land where one tick 600 s later lands, and both
    on the closed form: decay of the stored value by 600 s only."""
    many = _store_with(_concept())
    one = _store_with(_concept())

    _run_ticks(many, mode="since_last", start=T0, step_s=30, n=20)
    _run_ticks(one, mode="since_last", start=T0 + timedelta(seconds=570), step_s=30, n=1)

    # First-ever tick has no stamp: anchor = observed_at (23 h before T0), so it
    # applies the 23 h of decay the stored value never received, once.
    first_anchor_elapsed = 23 * 3600 + 600
    expected = 0.5 ** (first_anchor_elapsed / HL)
    assert _activation(many, "sub-concept-seed-test") == pytest.approx(expected, abs=1e-9)
    assert _activation(one, "sub-concept-seed-test") == pytest.approx(expected, abs=1e-9)


def test_since_last_is_far_from_the_legacy_compounding_cliff() -> None:
    """The live symptom: 1.0 -> 0.978 -> 0.956 -> 0.934 on consecutive 30 s ticks
    for a seed 23 h past observed_at. since_last must lose ~0.0008% per tick."""
    legacy = _store_with(_concept())
    fixed = _store_with(_concept(metadata={ACTIVATION_DECAYED_AT_KEY: T0.isoformat()}))

    _run_ticks(legacy, mode="legacy", start=T0, step_s=30, n=3)
    _run_ticks(fixed, mode="since_last", start=T0, step_s=30, n=3)

    assert _activation(legacy, "sub-concept-seed-test") == pytest.approx(0.978 ** 3, abs=2e-3)
    assert _activation(fixed, "sub-concept-seed-test") == pytest.approx(0.5 ** (90 / HL), abs=1e-9)


def test_legacy_mode_reproduces_old_behavior_exactly() -> None:
    store = _store_with(_concept())
    _run_ticks(store, mode="legacy", start=T0, step_s=30, n=5)

    observed = T0 - timedelta(hours=23)
    value = 1.0
    for i in range(1, 6):
        age = (T0 + timedelta(seconds=30 * i) - observed).total_seconds()
        value = round(decay_activation(current=value, elapsed_seconds=age, half_life_seconds=HL, floor=0.0), 6)
    assert _activation(store, "sub-concept-seed-test") == value
    # legacy never writes the stamp
    assert ACTIVATION_DECAYED_AT_KEY not in store.get_node_by_id("sub-concept-seed-test").metadata


# --- stamp semantics ---------------------------------------------------------


def test_missing_stamp_falls_back_to_observed_at_then_stamps() -> None:
    node = _concept(activation=0.8, observed_at=T0 - timedelta(hours=1), half_life=3600)
    store = _store_with(node)

    SubstrateDynamicsEngine(store=store).tick(now=T0)

    updated = store.get_node_by_id(node.node_id)
    assert updated.signals.activation.activation == pytest.approx(0.4, abs=1e-9)
    assert updated.metadata[ACTIVATION_DECAYED_AT_KEY] == T0.isoformat()


def test_garbled_stamp_falls_back_to_observed_at() -> None:
    node = _concept(observed_at=T0 - timedelta(hours=1), metadata={ACTIVATION_DECAYED_AT_KEY: "not-a-date"})
    assert activation_decay_anchor(node) == T0 - timedelta(hours=1)


def test_clock_going_backwards_neither_decays_nor_rewinds_the_stamp() -> None:
    node = _concept(activation=0.6, half_life=3600, metadata={ACTIVATION_DECAYED_AT_KEY: T0.isoformat()})
    store = _store_with(node)
    engine = SubstrateDynamicsEngine(store=store)

    engine.tick(now=T0 - timedelta(seconds=60))
    after_back = store.get_node_by_id(node.node_id)
    assert after_back.signals.activation.activation == 0.6
    assert after_back.metadata[ACTIVATION_DECAYED_AT_KEY] == T0.isoformat()

    engine.tick(now=T0 + timedelta(seconds=30))
    # Only the 30 s after the stamp count -- the backwards hop added nothing.
    assert _activation(store, node.node_id) == pytest.approx(0.6 * 0.5 ** (30 / 3600), abs=1e-12)


def test_node_rewritten_after_stamp_decays_from_new_observed_at() -> None:
    """A producer re-observes the node between ticks (newer observed_at, fresh
    activation) while the old stamp survives in metadata (Falkor preserves an
    omitted stamp; merge_node keeps existing metadata). Decay must run from the
    new observation, not from the older stamp."""
    store = _store_with(_concept(activation=1.0, half_life=3600, metadata={ACTIVATION_DECAYED_AT_KEY: T0.isoformat()}))
    engine = SubstrateDynamicsEngine(store=store)
    engine.tick(now=T0 + timedelta(seconds=30))
    stamped = store.get_node_by_id("sub-concept-seed-test")

    reobserved_at = T0 + timedelta(seconds=100)
    rewritten = stamped.model_copy(
        update={
            "temporal": stamped.temporal.model_copy(update={"observed_at": reobserved_at}),
            "signals": stamped.signals.model_copy(
                update={"activation": stamped.signals.activation.model_copy(update={"activation": 0.9})}
            ),
        }
    )
    store.upsert_node(identity_key="id:sub-concept-seed-test", node=rewritten)

    engine.tick(now=T0 + timedelta(seconds=130))
    updated = store.get_node_by_id("sub-concept-seed-test")
    assert updated.signals.activation.activation == pytest.approx(0.9 * 0.5 ** (30 / 3600), abs=1e-9)
    assert updated.metadata[ACTIVATION_DECAYED_AT_KEY] == (T0 + timedelta(seconds=130)).isoformat()


def test_write_guard_skip_keeps_stamp_and_value_together() -> None:
    """A low-activation node's per-tick decay is below the 1e-6 write threshold,
    so most ticks skip the write. The skipped tick must not advance the stamp,
    so the next write covers the full interval since the last real decay."""
    node = _concept(activation=0.05, metadata={ACTIVATION_DECAYED_AT_KEY: T0.isoformat()})
    store = _store_with(node)
    _run_ticks(store, mode="since_last", start=T0, step_s=30, n=40)

    assert store.upserts < 40  # the guard really skipped some ticks
    expected = 0.05 * 0.5 ** (1200 / HL)
    # Persisted value lags the true one by at most one sub-threshold step.
    assert abs(_activation(store, node.node_id) - expected) < 1e-6
    stamp = datetime.fromisoformat(store.get_node_by_id(node.node_id).metadata[ACTIVATION_DECAYED_AT_KEY])
    stored = _activation(store, node.node_id)
    assert stored == pytest.approx(0.05 * 0.5 ** ((stamp - T0).total_seconds() / HL), abs=1e-12)


def test_pressure_only_writes_do_not_freeze_decay() -> None:
    """A node rewritten every tick for pressure alone (activation change below
    threshold) must keep its old value+stamp pair; otherwise each write would
    round the sub-threshold decay away and restart the clock, freezing decay."""
    node = _concept(activation=0.05, metadata={ACTIVATION_DECAYED_AT_KEY: T0.isoformat()})
    store = _store_with(node)
    engine = SubstrateDynamicsEngine(store=store)
    flip = {"v": 0.0}

    def _alternating_pressures(nodes, outgoing, now):
        flip["v"] = 0.1 if flip["v"] == 0.0 else 0.0
        return {node.node_id: flip["v"]}, {node.node_id: "test"}

    engine._compute_pressures = _alternating_pressures  # type: ignore[method-assign]
    for i in range(1, 41):
        engine.tick(now=T0 + timedelta(seconds=30 * i))

    assert store.upserts == 40  # every tick wrote (pressure moved)
    expected = 0.05 * 0.5 ** (1200 / HL)
    assert abs(_activation(store, node.node_id) - expected) < 1e-6
    assert _activation(store, node.node_id) < 0.05


def test_fresh_seed_still_decays_by_age_not_since_last() -> None:
    """Seed input (salience) is recomputed every tick, so decaying it by full age
    is a closed form. It must not be treated as stored and escape decay."""
    observed = T0 - timedelta(days=30)
    node = _concept(activation=0.0, salience=1.0, observed_at=observed, half_life=HL)
    store = _store_with(node)
    SubstrateDynamicsEngine(store=store).tick(now=T0)
    # seed = 0.35 * salience (recency 0 at 30 d); one half-life of age.
    assert _activation(store, node.node_id) == pytest.approx(0.35 * 0.5, abs=1e-9)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_decay_mode("since-last")
    with pytest.raises(ValueError):
        SubstrateDynamicsEngine(store=InMemorySubstrateGraphStore(), decay_mode="bogus")
    assert normalize_decay_mode(None) == "since_last"
    assert normalize_decay_mode(" LEGACY ") == "legacy"


# --- durable round trip ------------------------------------------------------


def test_stamp_round_trips_through_falkor_codec_for_every_durable_kind() -> None:
    stamp = T0.isoformat()
    concept = _concept(metadata={ACTIVATION_DECAYED_AT_KEY: stamp})
    evidence = EvidenceNodeV1(
        node_id="ev-1",
        anchor_scope="orion",
        evidence_type="turn",
        content_ref="ref:1",
        temporal={"observed_at": T0},
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="test", source_channel="test", producer="test"
        ),
        metadata={ACTIVATION_DECAYED_AT_KEY: stamp},
    )
    for node in (concept, evidence):
        props = encode_node_properties(node, "id")
        assert props[ACTIVATION_DECAYED_AT_KEY] == stamp
        row = {key: props.get(key) for key in NATIVE_NODE_RETURN_FIELDS}
        decoded = decode_node(row)
        assert decoded.metadata[ACTIVATION_DECAYED_AT_KEY] == stamp
    assert ACTIVATION_DECAYED_AT_KEY in DYNAMICS_ENGINE_OWNED_METADATA_KEYS


def test_stamp_unaware_writer_is_stamped_with_its_own_observed_at() -> None:
    """concept_induction's blind re-save writes a fresh model with no stamp and
    an old observed_at. Its activation must be treated as valid as of that
    observation (decayed by real age next tick), not as fresh as of the newer
    durable stamp -- which would pin a re-saved seed at 1.0 indefinitely."""
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    store.upsert_node(identity_key="id:a", node=_concept(node_id="a", metadata={ACTIVATION_DECAYED_AT_KEY: T0.isoformat()}))
    assert client.calls[-1][1][ACTIVATION_DECAYED_AT_KEY] == T0.isoformat()

    resaved = _concept(node_id="a")  # observed_at = T0 - 23h, no stamp
    store.upsert_node(identity_key="id:a", node=resaved)
    cypher, params = client.calls[-1]
    expected = resaved.temporal.observed_at.isoformat()
    assert f"n.{ACTIVATION_DECAYED_AT_KEY} = ${ACTIVATION_DECAYED_AT_KEY}" in cypher
    assert params[ACTIVATION_DECAYED_AT_KEY] == expected
    cached = store.get_node_by_id("a")
    assert cached.metadata[ACTIVATION_DECAYED_AT_KEY] == expected
    assert activation_decay_anchor(cached) == resaved.temporal.observed_at


def test_codec_omits_absent_stamp_rather_than_writing_null() -> None:
    props = encode_node_properties(_concept(), "id")
    assert ACTIVATION_DECAYED_AT_KEY not in props


def test_pressure_only_write_pairs_value_with_its_own_stamp() -> None:
    """Race from review: this tick's snapshot holds (v0, s0); another writer
    (Hub scheduler) has since stored a newer, lower value with a newer stamp.
    A pressure-only write of v0 must carry s0, never inherit the newer stamp."""
    s0 = T0
    node = _concept(activation=0.05, metadata={ACTIVATION_DECAYED_AT_KEY: s0.isoformat()})
    store = _store_with(node)
    engine = SubstrateDynamicsEngine(store=store)
    engine._compute_pressures = lambda nodes, outgoing, now: ({node.node_id: 0.1}, {node.node_id: "t"})  # type: ignore[method-assign]
    engine.tick(now=T0 + timedelta(seconds=30))

    written = store.get_node_by_id(node.node_id)
    assert written.signals.activation.activation == 0.05
    assert written.metadata[ACTIVATION_DECAYED_AT_KEY] == s0.isoformat()


def test_legacy_mode_drops_stale_stamp_so_roll_forward_does_not_double_decay() -> None:
    node = _concept(metadata={ACTIVATION_DECAYED_AT_KEY: (T0 - timedelta(hours=5)).isoformat()})
    store = _store_with(node)
    SubstrateDynamicsEngine(store=store, decay_mode="legacy").tick(now=T0)
    assert ACTIVATION_DECAYED_AT_KEY not in store.get_node_by_id(node.node_id).metadata


def test_stamp_survives_the_metadata_key_cap() -> None:
    """A node already at the 16-key metadata cap must not lose its stamp to the
    sanitizer (it would be re-filled from observed_at and compound again)."""
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=RecordingFalkorClient(),
        hydrate=False,
    )
    metadata = {f"k{i}": i for i in range(20)}
    metadata[ACTIVATION_DECAYED_AT_KEY] = T0.isoformat()  # inserted last
    store.upsert_node(identity_key="id:full", node=_concept(node_id="full", metadata=metadata))
    assert store.get_node_by_id("full").metadata[ACTIVATION_DECAYED_AT_KEY] == T0.isoformat()
