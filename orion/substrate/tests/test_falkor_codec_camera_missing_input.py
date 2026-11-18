"""Round-trip of the camera surprise nodes' missing-input marker and embedding
clock through the FalkorDB codec (review finding 2026-10-21: both keys were
outside the closed allowlist, and upsert_node caches the decoded durable row,
so they vanished even in the writer's own process)."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.adapters._common import make_temporal
from orion.substrate.falkor_codec import (
    DYNAMICS_METADATA_KEYS,
    EXTERNALLY_OWNED_METADATA_KEYS,
    decode_node,
    encode_node_properties,
)


def _node(metadata: dict) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id="node:substrate.perception",
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="Perception prediction error",
        temporal=make_temporal(observed_at=datetime(2026, 10, 21, tzinfo=timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="substrate_prediction_error",
            source_channel="substrate.runtime", producer="substrate_runtime_worker", tier_rank=2,
        ),
        signals=SubstrateSignalBundleV1(confidence=1.0, salience=0.0),
        metadata=metadata,
    )


def _round_trip(metadata: dict) -> dict:
    row = encode_node_properties(_node(metadata), identity_key="k")
    return dict(decode_node(row).metadata)


def test_keys_are_allowlisted_and_externally_owned() -> None:
    for key in ("embedding_staleness", "prediction_error_missing_reason"):
        assert key in DYNAMICS_METADATA_KEYS
        assert key in EXTERNALLY_OWNED_METADATA_KEYS


def test_marker_and_staleness_round_trip() -> None:
    md = _round_trip({"prediction_error": 0.0, "embedding_staleness": 1.0,
                      "prediction_error_missing_reason": "stale"})
    assert md["prediction_error_missing_reason"] == "stale"
    assert md["embedding_staleness"] == 1.0
    assert md["prediction_error"] == 0.0


def test_real_reading_encodes_a_null_marker_so_on_match_clears_it() -> None:
    row = encode_node_properties(_node({"prediction_error": 0.3, "embedding_staleness": 0.0}), identity_key="k")
    assert "prediction_error_missing_reason" in row and row["prediction_error_missing_reason"] is None
    assert row["embedding_staleness"] == 0.0  # a healthy 0.0 is kept, not dropped
    md = dict(decode_node(row).metadata)
    assert "prediction_error_missing_reason" not in md
