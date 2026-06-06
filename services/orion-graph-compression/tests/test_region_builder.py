import pytest
from datetime import datetime, timezone


def test_region_builder_produces_valid_region():
    from app.clustering.region_builder import build_region, stable_region_id

    nodes = {"http://A", "http://B", "http://C"}
    region = build_region(
        nodes=nodes,
        scope="episodic",
        kind="community",
        summary="Test community summary.",
        summary_kind="structural",
        salience=0.6,
        trust_tier="unverified",
        compression_version="1.0.0",
    )
    assert region.scope == "episodic"
    assert region.kind == "community"
    assert region.summary == "Test community summary."
    assert len(region.exemplar_ids) > 0
    assert len(region.derived_from) > 0
    assert region.stale is False


def test_region_builder_stable_id_idempotent():
    from app.clustering.region_builder import stable_region_id

    nodes = frozenset({"http://A", "http://B"})
    id1 = stable_region_id(scope="episodic", kind="community", nodes=nodes)
    id2 = stable_region_id(scope="episodic", kind="community", nodes=nodes)
    assert id1 == id2
    assert id1.startswith("urn:orion:compression:region:")


def test_region_builder_different_nodes_different_id():
    from app.clustering.region_builder import stable_region_id

    id1 = stable_region_id("episodic", "community", frozenset({"http://A"}))
    id2 = stable_region_id("episodic", "community", frozenset({"http://B"}))
    assert id1 != id2


def test_region_builder_trust_tier_inherits_lowest():
    from app.clustering.region_builder import build_region

    region = build_region(
        nodes={"http://A"},
        scope="substrate",
        kind="contradiction",
        summary="Conflict found.",
        summary_kind="structural",
        salience=0.9,
        trust_tier="unverified",
        compression_version="1.0.0",
    )
    assert region.trust_tier == "unverified"
