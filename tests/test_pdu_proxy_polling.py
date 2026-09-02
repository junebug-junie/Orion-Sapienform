"""The hub measures circe's power on its behalf, because circe's NIC is dead.

Per-node PDU polling (#1690) was the right default -- it keeps `chassis_watts` a self-report.
It does not work for the one node the feature was built for. Confirmed live 2026-08-16, every
65 seconds:

    orion.biometrics.pdu - WARNING - pdu_poll_failed error=5 second timeout exceeded on UDP

circe reaches the bus over Tailscale and has no LAN path to the PDU. athena has both.

The load-bearing property here is **provenance**. `chassis_watts` has been a self-report, and a
proxy reading breaks that; without saying so, a future reader finding circe with a chassis
figure would reasonably conclude its BMC came back. So a proxied value is labelled, and it only
ever fills a gap -- it never overwrites what a node measured itself.
"""
from __future__ import annotations

import pytest

from app.pdu import parse_proxy_outlets
from orion.schemas.telemetry.biometrics import BiometricsClusterV1
from orion.telemetry.biometrics_pipeline import aggregate_fleet_measurements


def _merge(per_node, proxy):
    """The hub's merge, mirroring publish_cluster: proxy fills gaps only."""
    per_node = dict(per_node)
    proxied = {}
    for node, values in proxy.items():
        own = per_node.get(node) or {}
        filled = {k: v for k, v in values.items() if own.get(k) is None}
        if not filled:
            continue
        per_node[node] = {**own, **filled}
        proxied[node] = sorted(filled)
    return per_node, proxied


# ------------------------------------------------------------ the gap it closes

def test_circe_stops_being_missing_from_the_fleet_total():
    """The point. Before: chassis_watts absent, circe named in measurements_missing forever."""
    per_node = {
        "athena": {"chassis_watts": 470.0},
        "atlas": {"chassis_watts": 291.0, "pdu_watts": 266.0},
        "circe": {"gpu_watts_total": 153.0},          # no chassis -- dead NIC, no BMC
    }
    merged, proxied = _merge(per_node, {"circe": {"chassis_watts": 419.0, "pdu_watts": 419.0}})
    totals, missing = aggregate_fleet_measurements(merged)

    assert totals["chassis_watts"] == pytest.approx(1180.0)
    assert "chassis_watts" not in missing
    assert proxied["circe"] == ["chassis_watts", "pdu_watts"]


def test_a_powered_down_node_reads_a_true_zero_rather_than_vanishing():
    """Strictly better than self-polling: the outlets report circe whether it is up or not.
    0 W is a real reading and belongs in the total; absent would understate the fleet."""
    merged, proxied = _merge({"circe": {}}, {"circe": {"chassis_watts": 0.0, "pdu_watts": 0.0}})
    totals, missing = aggregate_fleet_measurements(merged)
    assert totals["chassis_watts"] == 0.0
    assert "chassis_watts" not in missing
    assert proxied["circe"] == ["chassis_watts", "pdu_watts"]


# ------------------------------------------------------------ per-node breakdown survives the sum

def test_the_merged_per_node_dict_is_what_the_cluster_schema_should_carry_by_node():
    """publish_cluster() builds this exact `merged` dict, then only its SUM
    (aggregate_fleet_measurements) used to make it into BiometricsClusterV1 -- losing which
    machine drew how much. measurements_by_node exists so this dict (proxy-filled, self-report
    still winning) rides along per-node instead of being discarded after summing."""
    per_node = {
        "athena": {"chassis_watts": 470.0},
        "circe": {"gpu_watts_total": 153.0},
    }
    merged, _ = _merge(per_node, {"circe": {"chassis_watts": 419.0, "pdu_watts": 419.0}})
    c = BiometricsClusterV1(sources=["athena", "circe"], measurements_by_node=merged)
    rt = BiometricsClusterV1.model_validate_json(c.model_dump_json())
    # athena's self-report and circe's proxied reading are both individually recoverable --
    # the whole point, since aggregate_fleet_measurements' total alone cannot tell them apart.
    assert rt.measurements_by_node["athena"]["chassis_watts"] == pytest.approx(470.0)
    assert rt.measurements_by_node["circe"]["chassis_watts"] == pytest.approx(419.0)


# ------------------------------------------------------------ provenance

def test_a_proxied_reading_is_labelled():
    """Without this, circe having chassis_watts reads as 'the BMC came back'."""
    _, proxied = _merge({"circe": {}}, {"circe": {"chassis_watts": 419.0}})
    assert proxied == {"circe": ["chassis_watts"]}


def test_a_self_report_always_wins_and_is_not_labelled_as_proxied():
    """atlas measures itself. The hub must not overwrite that, and must not claim it did."""
    merged, proxied = _merge(
        {"atlas": {"chassis_watts": 291.0}},
        {"atlas": {"chassis_watts": 266.0}},
    )
    assert merged["atlas"]["chassis_watts"] == 291.0   # self-report kept
    assert proxied == {}                                # nothing was filled


def test_a_proxy_fills_only_the_missing_keys():
    """circe self-reports GPU watts but not chassis. Only chassis is proxied."""
    merged, proxied = _merge(
        {"circe": {"gpu_watts_total": 153.0}},
        {"circe": {"chassis_watts": 419.0, "gpu_watts_total": 999.0}},
    )
    assert merged["circe"]["gpu_watts_total"] == 153.0   # untouched
    assert merged["circe"]["chassis_watts"] == 419.0
    assert proxied["circe"] == ["chassis_watts"]


def test_a_zero_self_report_is_a_reading_and_blocks_the_proxy():
    """0.0 W measured by the node is data, not absence. `own.get(k) is None` is the test, and
    a truthiness check here would let a proxy silently replace a real zero."""
    merged, proxied = _merge({"circe": {"chassis_watts": 0.0}},
                             {"circe": {"chassis_watts": 419.0}})
    assert merged["circe"]["chassis_watts"] == 0.0
    assert proxied == {}


def test_the_cluster_schema_carries_provenance():
    c = BiometricsClusterV1(
        sources=["athena", "atlas", "circe"],
        measurements={"chassis_watts": 1180.0},
        measurements_proxied={"circe": ["chassis_watts"]},
    )
    rt = BiometricsClusterV1.model_validate_json(c.model_dump_json())
    assert rt.measurements_proxied == {"circe": ["chassis_watts"]}


def test_producers_predating_this_report_none():
    """None means 'this producer does not report provenance', NOT 'nothing was proxied'."""
    assert BiometricsClusterV1.model_validate({"sources": ["athena"]}).measurements_proxied is None


# ------------------------------------------------------------ a broken proxy is not a reading

def test_a_failed_proxy_leaves_the_node_missing():
    """A proxy that is itself broken must not look like a measurement -- otherwise the fix for
    silent failure becomes a new way to fail silently."""
    merged, proxied = _merge({"circe": {}}, {})     # poller returned nothing
    totals, missing = aggregate_fleet_measurements({**merged, "athena": {"chassis_watts": 470.0}})
    assert proxied == {}
    assert missing["chassis_watts"] == ["circe"]


# ------------------------------------------------------------ config parsing

def test_the_proxy_map_parses():
    assert parse_proxy_outlets('{"circe": [19,25,31]}') == {"circe": [19, 25, 31]}
    assert parse_proxy_outlets('{"circe": "19,25,31"}') == {"circe": [19, 25, 31]}


def test_node_names_are_normalised():
    assert parse_proxy_outlets('{" Circe ": [19]}') == {"circe": [19]}


def test_an_empty_or_broken_map_disables_proxying_rather_than_crashing():
    for raw in ("", "   ", "not json", "[1,2,3]", '{"circe": null}', '{"circe": []}'):
        assert parse_proxy_outlets(raw) == {}


def test_garbage_outlets_inside_a_valid_map_are_dropped():
    assert parse_proxy_outlets('{"circe": [19, "banana", 0, 31]}') == {"circe": [19, 31]}


def test_multiple_proxied_nodes_are_supported():
    assert parse_proxy_outlets('{"circe": [19,25], "ghost": [7]}') == {
        "circe": [19, 25], "ghost": [7]}
