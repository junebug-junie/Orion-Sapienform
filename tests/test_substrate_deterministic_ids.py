from __future__ import annotations

from orion.substrate.ids import stable_delta_id, stable_hash_id, stable_receipt_id


def test_stable_hash_id_is_deterministic() -> None:
    a = stable_hash_id("rcpt", ["reducer", "gev_1", "gev_2"])
    b = stable_hash_id("rcpt", ["reducer", "gev_1", "gev_2"])
    assert a == b
    assert a.startswith("rcpt_")


def test_stable_receipt_id_sorts_event_buckets() -> None:
    r1 = stable_receipt_id(
        reducer_id="node_pressure_reducer",
        accepted_event_ids=["gev_b", "gev_a"],
        rejected_event_ids=[],
        merged_event_ids=[],
        noop_event_ids=[],
        emission_id="oem_1",
    )
    r2 = stable_receipt_id(
        reducer_id="node_pressure_reducer",
        accepted_event_ids=["gev_a", "gev_b"],
        rejected_event_ids=[],
        merged_event_ids=[],
        noop_event_ids=[],
        emission_id="oem_1",
    )
    assert r1 == r2


def test_stable_delta_id_normalizes_node_id() -> None:
    d1 = stable_delta_id(
        reducer_id="biometrics_node_reducer",
        target_projection="proj_node_bio",
        target_kind="node_biometrics",
        target_id="Atlas",
        operation="update",
        caused_by_event_ids=["gev_1"],
    )
    d2 = stable_delta_id(
        reducer_id="biometrics_node_reducer",
        target_projection="proj_node_bio",
        target_kind="node_biometrics",
        target_id="atlas",
        operation="update",
        caused_by_event_ids=["gev_1"],
    )
    assert d1 == d2
    assert d1.startswith("delta_")
