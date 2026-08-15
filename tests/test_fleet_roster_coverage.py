"""A fleet total can never quietly shed a whole machine.

`measurements_missing` closes the gap for a node that reported but lacked a key. It cannot
close the gap for a node that stopped reporting at all: such a node never enters `sources`,
so it leaves no trace and the totals read as complete.

Observed live 2026-08-14 while circe was down:

    chassis_watts: 646.0        measurements_missing: null

646 W is a two-machine sum presented as the whole fleet, with nothing anywhere saying so.
"""
from __future__ import annotations

import pytest

from orion.biometrics.node_catalog import NodeCatalog, NodeProfile
from orion.schemas.telemetry.biometrics import BiometricsClusterV1


def _catalog(tmp_path, nodes):
    p = tmp_path / "node_catalog.yaml"
    body = ["defaults:", "  role: unknown", "  expected_online: true", "nodes:"]
    for n in nodes:
        body += [f"  {n}:", "    role: burst_gpu", "    expected_online: true"]
    p.write_text("\n".join(body) + "\n")
    return NodeCatalog.load(p)


def _absent(catalog: NodeCatalog, reporting) -> list:
    """The computation under test, as publish_cluster performs it."""
    contributed = {catalog.resolve(n).node_id for n in reporting}
    return sorted(nid for nid in catalog.profiles if nid not in contributed)


# --------------------------------------------------------------- the invariant

def test_a_silent_node_is_named(tmp_path):
    """The live case. athena and atlas reporting, circe down."""
    cat = _catalog(tmp_path, ["athena", "atlas", "circe"])
    assert _absent(cat, ["athena", "atlas"]) == ["circe"]


def test_a_complete_fleet_reports_nobody_absent(tmp_path):
    cat = _catalog(tmp_path, ["athena", "atlas", "circe"])
    assert _absent(cat, ["athena", "atlas", "circe"]) == []


def test_every_node_silent_names_every_node(tmp_path):
    """A hub that has heard from nobody must not publish an empty-but-confident cluster."""
    cat = _catalog(tmp_path, ["athena", "atlas", "circe"])
    assert _absent(cat, []) == ["athena", "atlas", "circe"]


def test_absence_is_resolved_through_aliases_not_raw_strings(tmp_path):
    """Nodes publish under names like `circe.tail348bbe.ts.net`. Comparing raw strings against
    canonical ids would report a node as absent while it is actively reporting."""
    p = tmp_path / "node_catalog.yaml"
    p.write_text(
        "defaults:\n  role: unknown\n  expected_online: true\n"
        "nodes:\n"
        "  circe:\n    role: burst_gpu\n    expected_online: true\n"
        "    aliases:\n      - circe.tail348bbe.ts.net\n"
        "  athena:\n    role: hub\n    expected_online: true\n"
    )
    cat = NodeCatalog.load(p)
    assert _absent(cat, ["circe.tail348bbe.ts.net", "athena"]) == []


def test_an_unknown_node_reporting_does_not_mask_a_known_one(tmp_path):
    """A node not in the catalog resolves to known=False with its own id. It must not
    accidentally satisfy the roster for a catalogued node that is genuinely silent."""
    cat = _catalog(tmp_path, ["athena", "circe"])
    assert _absent(cat, ["athena", "some-new-box"]) == ["circe"]


# --------------------------------------------------------------- schema carriage

def test_the_cluster_carries_absence_alongside_a_partial_total():
    """The two coverage fields answer different questions and both are needed: circe absent
    entirely, atlas present but without a BMC."""
    c = BiometricsClusterV1(
        sources=["athena", "atlas"],
        nodes_absent=["circe"],
        measurements={"chassis_watts": 646.0},
        measurements_missing={"fan_pct_max": ["atlas"]},
    )
    rt = BiometricsClusterV1.model_validate_json(c.model_dump_json())
    assert rt.nodes_absent == ["circe"]
    assert rt.measurements["chassis_watts"] == 646.0
    assert rt.measurements_missing["fan_pct_max"] == ["atlas"]


def test_absence_defaults_to_none_for_producers_predating_this():
    """None means 'this producer does not report coverage', NOT 'nobody is absent'. A consumer
    must not read a missing field as a completeness guarantee."""
    c = BiometricsClusterV1.model_validate({"sources": ["athena"]})
    assert c.nodes_absent is None


def test_none_and_empty_list_are_distinguishable():
    """[] is a current producer saying 'the whole roster reported'. That is a real claim and
    has to survive a round trip distinctly from None."""
    assert BiometricsClusterV1(nodes_absent=[]).nodes_absent == []
    assert BiometricsClusterV1().nodes_absent is None


# --------------------------------------------------------------- what it deliberately is NOT

def test_absence_carries_no_verdict_about_whether_it_is_a_problem():
    """circe is run intermittently to save cost, so its silence is expected; the same silence
    from athena would be an outage. The aggregate states WHO is absent and nothing more --
    `expected_online` is already committed to a different job (gating whether pressure_organ
    suppresses a node's strain as expected staleness), so it cannot settle this either."""
    fields = set(BiometricsClusterV1.model_fields)
    for verdict in ("nodes_unexpectedly_silent", "roster_ok", "degraded", "alarm"):
        assert verdict not in fields
    assert "nodes_absent" in fields


def test_a_node_profile_still_exposes_expected_online_untouched(tmp_path):
    """Regression guard: this patch must not repurpose the flag it declined to repurpose."""
    cat = _catalog(tmp_path, ["circe"])
    profile = cat.resolve("circe")
    assert isinstance(profile, NodeProfile)
    assert profile.expected_online is True
