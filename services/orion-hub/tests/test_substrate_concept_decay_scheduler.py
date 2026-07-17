from __future__ import annotations

import os
import sys
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from scripts import api_routes  # noqa: E402

from orion.core.schemas.cognitive_substrate import (  # noqa: E402
    ConceptNodeV1,
    EntityNodeV1,
    SubstrateActivationV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.adapters._common import make_provenance, make_temporal  # noqa: E402
from orion.substrate.store import InMemorySubstrateGraphStore  # noqa: E402


def _make_concept_node(
    *,
    node_id: str,
    activation: float,
    decay_half_life_seconds: int | None,
    decay_floor: float,
    observed_at: datetime,
) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        anchor_scope="world",
        label=f"concept-{node_id}",
        temporal=make_temporal(observed_at=observed_at),
        signals=SubstrateSignalBundleV1(
            activation=SubstrateActivationV1(
                activation=activation,
                recency_score=0.5,
                decay_half_life_seconds=decay_half_life_seconds,
                decay_floor=decay_floor,
            )
        ),
        provenance=make_provenance(
            source_kind="test.fixture",
            source_channel="test:substrate:decay",
            producer="test_substrate_concept_decay_scheduler",
        ),
    )


def test_decay_concept_activations_decays_old_node_toward_floor(monkeypatch) -> None:
    fresh_store = InMemorySubstrateGraphStore()
    old_observed_at = datetime.now(timezone.utc) - timedelta(hours=6)
    node = _make_concept_node(
        node_id="concept-old",
        activation=1.0,
        decay_half_life_seconds=3600,  # 1 hour half-life, 6 hours elapsed -> heavy decay
        decay_floor=0.1,
        observed_at=old_observed_at,
    )
    fresh_store.upsert_node(identity_key=node.node_id, node=node)
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    summary = api_routes.decay_concept_activations()

    assert summary == {"decayed": 1, "skipped": 0, "errors": 0, "total_concepts": 1}
    updated = fresh_store.get_node_by_id("concept-old")
    assert updated is not None
    new_activation = updated.signals.activation.activation
    assert new_activation < 1.0
    # 6 hours / 1 hour half-life = 6 half-lives -> well below the original value,
    # but never below the floor.
    assert new_activation < 0.2
    assert new_activation >= 0.1


def test_decay_concept_activations_barely_moves_recent_node(monkeypatch) -> None:
    fresh_store = InMemorySubstrateGraphStore()
    recent_observed_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    node = _make_concept_node(
        node_id="concept-recent",
        activation=0.8,
        decay_half_life_seconds=3600,
        decay_floor=0.0,
        observed_at=recent_observed_at,
    )
    fresh_store.upsert_node(identity_key=node.node_id, node=node)
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    summary = api_routes.decay_concept_activations()

    assert summary == {"decayed": 1, "skipped": 0, "errors": 0, "total_concepts": 1}
    updated = fresh_store.get_node_by_id("concept-recent")
    assert updated is not None
    new_activation = updated.signals.activation.activation
    # Barely any time has passed -- activation should be close to the original.
    assert abs(new_activation - 0.8) < 0.01


def test_decay_concept_activations_only_touches_activation_field(monkeypatch) -> None:
    fresh_store = InMemorySubstrateGraphStore()
    node = _make_concept_node(
        node_id="concept-preserve",
        activation=0.9,
        decay_half_life_seconds=60,
        decay_floor=0.0,
        observed_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    node = node.model_copy(update={"promotion_state": "canonical", "metadata": {"foo": "bar"}})
    fresh_store.upsert_node(identity_key=node.node_id, node=node)
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    api_routes.decay_concept_activations()

    updated = fresh_store.get_node_by_id("concept-preserve")
    assert updated is not None
    assert updated.promotion_state == "canonical"
    assert updated.metadata == {"foo": "bar"}
    assert updated.label == "concept-concept-preserve"
    # Only the activation value changed -- recency_score/decay_half_life_seconds/decay_floor untouched.
    assert updated.signals.activation.recency_score == 0.5
    assert updated.signals.activation.decay_half_life_seconds == 60
    assert updated.signals.activation.decay_floor == 0.0


def test_decay_concept_activations_skips_malformed_node_without_crashing(monkeypatch) -> None:
    fresh_store = InMemorySubstrateGraphStore()
    good_node = _make_concept_node(
        node_id="concept-good",
        activation=0.7,
        decay_half_life_seconds=3600,
        decay_floor=0.0,
        observed_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )
    fresh_store.upsert_node(identity_key=good_node.node_id, node=good_node)

    # A non-concept node (entity) must not be touched or counted -- _concept_nodes-style
    # filtering by node_kind == "concept" should exclude it entirely.
    entity_node = EntityNodeV1(
        node_id="entity-1",
        anchor_scope="world",
        label="not-a-concept",
        temporal=make_temporal(observed_at=datetime.now(timezone.utc)),
        provenance=make_provenance(
            source_kind="test.fixture",
            source_channel="test:substrate:decay",
            producer="test_substrate_concept_decay_scheduler",
        ),
    )
    fresh_store.upsert_node(identity_key=entity_node.node_id, node=entity_node)

    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    summary = api_routes.decay_concept_activations()

    assert summary["total_concepts"] == 1
    assert summary["decayed"] == 1
    assert summary["skipped"] == 0
    assert summary["errors"] == 0
    # Entity node is untouched (not even inspected as a concept).
    entity_after = fresh_store.get_node_by_id("entity-1")
    assert entity_after is not None
    assert entity_after.label == "not-a-concept"


def test_decay_concept_activations_skips_malformed_activation_without_crashing(monkeypatch) -> None:
    """A concept node whose signals/temporal bundle is missing/malformed must be
    skipped (counted in ``skipped``), not crash the rest of the pass."""

    class _FakeSignals:
        # No `.activation` attribute at all -- simulates a malformed bundle.
        pass

    class _MalformedConceptNode:
        node_id = "concept-malformed"
        node_kind = "concept"
        signals = _FakeSignals()
        temporal = None

    fresh_store = InMemorySubstrateGraphStore()
    good_node = _make_concept_node(
        node_id="concept-good-2",
        activation=0.6,
        decay_half_life_seconds=3600,
        decay_floor=0.0,
        observed_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    fresh_store.upsert_node(identity_key=good_node.node_id, node=good_node)
    # Bypass pydantic validation entirely by writing directly into the store's
    # internal node dict -- this is the only way to construct a genuinely
    # malformed node, since ConceptNodeV1 itself would reject one.
    fresh_store._nodes[_MalformedConceptNode.node_id] = _MalformedConceptNode()

    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    summary = api_routes.decay_concept_activations()

    assert summary["total_concepts"] == 2
    assert summary["decayed"] == 1
    assert summary["skipped"] == 1
    assert summary["errors"] == 0
    # The good node was still processed correctly despite the malformed sibling.
    updated_good = fresh_store.get_node_by_id("concept-good-2")
    assert updated_good.signals.activation.activation < 0.6


def test_decay_concept_activations_snapshot_failure_never_raises(monkeypatch) -> None:
    class _BoomStore:
        def snapshot(self):
            raise RuntimeError("store is unreachable")

    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", _BoomStore())

    summary = api_routes.decay_concept_activations()

    assert summary["decayed"] == 0
    assert summary["errors"] == 1
    assert summary["total_concepts"] == 0


def test_decay_concept_activations_empty_store_returns_clean_zero_summary(monkeypatch) -> None:
    fresh_store = InMemorySubstrateGraphStore()
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", fresh_store)

    summary = api_routes.decay_concept_activations()

    assert summary == {"decayed": 0, "skipped": 0, "errors": 0, "total_concepts": 0}


def test_decay_concept_activations_repeated_ticks_match_single_equivalent_call(monkeypatch) -> None:
    """Regression for the compounding-decay bug: calling decay_concept_activations
    N times with elapsed_seconds=tick (as the scheduler does every real tick) must
    land on the same activation as a single call with elapsed_seconds=N*tick --
    NOT the far-more-collapsed value you get from re-deriving elapsed from
    observed_at on every call without ever advancing the reference point."""
    half_life = 3600
    floor = 0.1
    start_activation = 1.0
    tick_seconds = 120.0
    ticks = 30  # 30 * 120s = 3600s = exactly one half-life

    repeated_store = InMemorySubstrateGraphStore()
    node = _make_concept_node(
        node_id="concept-repeated",
        activation=start_activation,
        decay_half_life_seconds=half_life,
        decay_floor=floor,
        observed_at=datetime.now(timezone.utc),
    )
    repeated_store.upsert_node(identity_key=node.node_id, node=node)
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", repeated_store)

    for _ in range(ticks):
        api_routes.decay_concept_activations(elapsed_seconds=tick_seconds)

    repeated_result = repeated_store.get_node_by_id("concept-repeated").signals.activation.activation

    single_call_store = InMemorySubstrateGraphStore()
    single_call_store.upsert_node(
        identity_key=node.node_id,
        node=_make_concept_node(
            node_id="concept-repeated",
            activation=start_activation,
            decay_half_life_seconds=half_life,
            decay_floor=floor,
            observed_at=datetime.now(timezone.utc),
        ),
    )
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", single_call_store)

    api_routes.decay_concept_activations(elapsed_seconds=ticks * tick_seconds)
    single_call_result = single_call_store.get_node_by_id("concept-repeated").signals.activation.activation

    assert abs(repeated_result - single_call_result) < 1e-9
    # decay_activation() is max(floor, current * 0.5**(elapsed/half_life)) -- after
    # exactly one half-life that's max(0.1, 1.0 * 0.5) == 0.5, not collapsed near floor.
    expected = max(floor, start_activation * 0.5)
    assert abs(repeated_result - expected) < 1e-6
