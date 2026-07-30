"""Regression guard (2026-07-30): SubstrateGraphMaterializer's merge-branch
upsert_node call must protect reducer-owned metadata keys (prediction_error,
contributing_turn_ids) the same way SubstrateDynamicsEngine.tick() already
does -- see falkor_codec.EXTERNALLY_OWNED_METADATA_KEYS's docstring for the
original incident this pattern was built for.

Confirmed live: orion-cortex-exec-background's concept-induction pipeline
re-materializes node:substrate.bus_synaptic (and any other reducer-owned
node whose anchor_scope/subject_ref/label happens to collide with an
induced concept's identity) every few seconds via this exact merge path --
far more often than the owning reducer's own tick cadence -- durably
re-locking a stale prediction_error read back into itself on every pass.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

from orion.core.schemas.concept_induction import ConceptEvidenceRef, ConceptItem, ConceptProfile
from orion.substrate import InMemorySubstrateGraphStore, SubstrateGraphMaterializer, map_concept_profile_to_substrate
from orion.substrate.falkor_codec import EXTERNALLY_OWNED_METADATA_KEYS


def _concept_profile(*, concept_id: str, label: str) -> ConceptProfile:
    now = datetime.now(timezone.utc)
    evidence = ConceptEvidenceRef(message_id=uuid4(), timestamp=now, channel="orion:test")
    return ConceptProfile(
        profile_id=f"profile-{concept_id}",
        subject="orion",
        window_start=now - timedelta(hours=1),
        window_end=now,
        concepts=[ConceptItem(concept_id=concept_id, label=label, confidence=0.8, salience=0.6, evidence=[evidence])],
        metadata={"subject_ref": "project:orion"},
    )


def test_apply_record_protects_reducer_owned_keys_on_merge_but_not_on_create() -> None:
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)

    record = map_concept_profile_to_substrate(profile=_concept_profile(concept_id="c-guard", label="Coherence"))

    with patch.object(store, "upsert_node", wraps=store.upsert_node) as spy:
        materializer.apply_record(record)  # first pass: create branch
        create_call = spy.call_args
        materializer.apply_record(record)  # second pass: merge branch (same identity)
        merge_call = spy.call_args

    # Create branch: no existing node to protect, must not pass skip_metadata_keys.
    assert create_call.kwargs.get("skip_metadata_keys") is None
    # Merge branch: must protect reducer-owned metadata from this materializer's
    # own write, regardless of what it read into canonical_node.metadata.
    assert merge_call.kwargs.get("skip_metadata_keys") == EXTERNALLY_OWNED_METADATA_KEYS
