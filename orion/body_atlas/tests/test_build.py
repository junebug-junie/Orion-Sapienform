from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.body_atlas.build import ObservedBus, build_atlas, fetch_observed

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
RECENT = NOW.timestamp() - 60
STALE = NOW.timestamp() - 30 * 86400


def _ch(name, producers, consumers):
    return {"name": name, "producer_services": producers, "consumer_services": consumers}


CATALOG = [
    _ch("orion:vision:frames", ["orion-vision-edge"], ["orion-vision-host"]),
    _ch("orion:vision:caption", ["orion-vision-host"], ["orion-vision-edge"]),
    _ch("orion:cortex:request", ["orion-cortex-orch"], ["orion-cortex-exec"]),
    _ch("orion:cortex:result", ["orion-cortex-exec"], ["orion-cortex-orch"]),
    _ch("orion:system:health", ["*"], ["orion-equilibrium-service"]),
    _ch("orion:bridge", ["orion-vision-host"], ["orion-cortex-orch"]),
]


def _by_name(atlas):
    return {p.name: p for p in atlas.parts}


def _flow(atlas, source, target):
    return next(f for f in atlas.flows if f.source == f"part:{source}" and f.target == f"part:{target}")


def test_short_bus_names_resolve_to_catalog_names():
    observed = ObservedBus(publishes=[("vision-edge", "orion:vision:frames", 500, RECENT)])
    atlas = build_atlas(CATALOG, observed, now=NOW)
    part = _by_name(atlas)["orion-vision-edge"]
    assert part.status == "live" and part.observed_ids == ["vision-edge"] and part.publish_count == 500
    assert "observed_publish" in _flow(atlas, "orion-vision-edge", "orion-vision-host").evidence


def test_transient_needs_low_count_and_stale_and_undeclared():
    observed = ObservedBus(publishes=[
        ("smoke:claude", "orion:cortex:request", 2, STALE),        # transient
        ("orion-situational", "orion:cortex:request", 29, RECENT),  # low but live: a real part
        ("security-watcher", "orion:cortex:request", 2_000_000, STALE),  # big: dormant part
        ("cortex-orch", "orion:cortex:request", 3, STALE),          # declared: never transient
    ])
    parts = _by_name(build_atlas(CATALOG, observed, now=NOW))
    assert parts["smoke:claude"].status == "transient" and parts["smoke:claude"].kind == "endpoint"
    assert parts["orion-situational"].status == "live"
    assert parts["security-watcher"].status == "dormant"
    assert parts["orion-cortex-orch"].status == "dormant"
    assert parts["orion-vision-host"].status == "declared_only"


def test_transient_parts_stay_out_of_organs_but_keep_their_flows():
    observed = ObservedBus(publishes=[("smoke:claude", "orion:cortex:request", 2, STALE)])
    atlas = build_atlas(CATALOG, observed, now=NOW)
    in_organs = {m for o in atlas.organs for m in o.members}
    assert "part:smoke:claude" not in in_organs
    assert _flow(atlas, "smoke:claude", "orion-cortex-exec")
    assert atlas.stats.transient_parts == 1


def test_broadcast_channel_contributes_one_unit_however_many_publishers():
    many = [(f"svc-{i}", "orion:system:health", 10_000, RECENT) for i in range(50)]
    atlas = build_atlas(CATALOG, ObservedBus(publishes=many), now=NOW)
    heartbeat = sum(f.weight for f in atlas.flows if f.channels == ["orion:system:health"])
    # the unit is log10(1 + all heartbeat messages), not 50 x log10(1 + 10,000)
    assert heartbeat == pytest.approx(__import__("math").log10(1 + 50 * 10_000), rel=1e-3)


def test_declared_channel_unit_is_split_over_its_pairs():
    catalog = [_ch("orion:fan", ["a", "b"], ["c", "d"])]
    atlas = build_atlas(catalog, None, now=NOW)
    assert sum(f.weight for f in atlas.flows) == pytest.approx(1.0)
    assert all(f.evidence == ["declared"] for f in atlas.flows)


def test_causal_edges_link_parts_with_their_own_evidence():
    observed = ObservedBus(causal=[("cortex-orch", "vision-host", 9, RECENT)])
    flow = _flow(build_atlas(CATALOG, observed, now=NOW), "orion-cortex-orch", "orion-vision-host")
    assert flow.evidence == ["observed_causal"] and flow.causal_count == 9


def test_organs_follow_circulation_and_report_modularity():
    atlas = build_atlas(CATALOG, None, now=NOW)
    groups = [set(o.members) for o in atlas.organs]
    assert {"part:orion-vision-edge", "part:orion-vision-host"} in groups
    assert {"part:orion-cortex-orch", "part:orion-cortex-exec"} in groups
    assert atlas.stats.modularity >= 0.3
    vision = next(o for o in atlas.organs if "part:orion-vision-edge" in o.members)
    assert vision.namespaces == ["vision"] and vision.name is None


def test_same_input_same_atlas_ids():
    a = build_atlas(CATALOG, None, now=NOW)
    b = build_atlas(list(reversed(CATALOG)), None, now=NOW)
    assert [o.organ_id for o in a.organs] == [o.organ_id for o in b.organs]
    assert a.model_dump() == b.model_dump()


def test_fetch_observed_accepts_string_valued_replies():
    class Fake:
        def graph_query(self, cypher):
            if "PUBLISHES" in cypher:
                return [{"source": "hub", "channel": "orion:x", "count": "12", "last_seen": "1792795604.8"}]
            return [{"source": "hub", "target": "thought", "count": 3.0, "last_seen": None}]

    obs = fetch_observed(Fake())
    assert obs.publishes == [("hub", "orion:x", 12, 1792795604.8)]
    assert obs.causal == [("hub", "thought", 3, 0.0)]
