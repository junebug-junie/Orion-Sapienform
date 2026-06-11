from __future__ import annotations

from orion.graph.rdf_retention import (
    GraphRetentionPolicy,
    build_artifact_cap_select,
    build_subject_age_delete,
    cutoff_literal,
    parse_retention_policies,
)


def test_parse_default_policies_when_unset():
    policies = parse_retention_policies(None)
    assert policies == []


def test_build_subject_age_delete_includes_cutoff():
    policy = GraphRetentionPolicy(
        graph="http://conjourney.net/graph/orion/chat",
        max_age_days=30,
    )
    sparql = build_subject_age_delete(policy, cutoff_literal(30))
    assert "GRAPH <http://conjourney.net/graph/orion/chat>" in sparql
    assert "DELETE" in sparql
    assert "orion#timestamp" in sparql


def test_build_artifact_cap_select_offsets():
    policy = GraphRetentionPolicy(
        graph="http://conjourney.net/graph/autonomy/drives",
        max_artifacts=3000,
    )
    sparql = build_artifact_cap_select(policy)
    assert "OFFSET 3000" in sparql
    assert "LIMIT 200" in sparql
    assert "AutonomyArtifact" in sparql


def test_parse_custom_json_policies():
    raw = '[{"graph":"http://example/graph/x","max_age_days":7}]'
    policies = parse_retention_policies(raw)
    assert len(policies) == 1
    assert policies[0].graph == "http://example/graph/x"
    assert policies[0].max_age_days == 7
