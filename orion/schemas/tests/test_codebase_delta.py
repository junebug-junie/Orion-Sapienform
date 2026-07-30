from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.codebase_delta import (
    CodebaseDeltaV1,
    GitDeltaPayloadV1,
    GraphDeltaPayloadV1,
    PrLifecycleDeltaPayloadV1,
)
from orion.schemas.registry import _REGISTRY, resolve

_NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)


def test_codebase_delta_registered() -> None:
    assert "CodebaseDeltaV1" in _REGISTRY
    assert resolve("CodebaseDeltaV1") is CodebaseDeltaV1


def test_git_domain_round_trip() -> None:
    event = CodebaseDeltaV1(
        domain="git",
        observed_at=_NOW,
        git=GitDeltaPayloadV1(
            prev_sha="a" * 40,
            head_sha="b" * 40,
            commit_count=3,
            files_added=1,
            files_deleted=0,
            files_modified=2,
            lines_added=100,
            lines_removed=40,
        ),
    )
    dumped = event.model_dump(mode="json")
    restored = CodebaseDeltaV1.model_validate(dumped)
    assert restored == event
    assert restored.pr_lifecycle is None
    assert restored.graph is None


def test_pr_lifecycle_domain_round_trip() -> None:
    event = CodebaseDeltaV1(
        domain="pr_lifecycle",
        observed_at=_NOW,
        pr_lifecycle=PrLifecycleDeltaPayloadV1(
            since=_NOW,
            until=_NOW,
            submitted_count=5,
            merged_count=3,
            closed_without_merge_count=0,
            submitted_numbers=[1, 2, 3, 4, 5],
            merged_numbers=[1, 2, 3],
            possibly_truncated=False,
        ),
    )
    dumped = event.model_dump(mode="json")
    restored = CodebaseDeltaV1.model_validate(dumped)
    assert restored == event
    assert restored.git is None
    assert restored.graph is None


def test_graph_domain_round_trip_with_none_jaccard() -> None:
    """god_node_jaccard_similarity=None (unparseable GRAPH_REPORT.md) must
    survive a JSON round-trip as None, not get coerced to 0.0/1.0."""
    event = CodebaseDeltaV1(
        domain="graph",
        observed_at=_NOW,
        graph=GraphDeltaPayloadV1(
            node_count_delta=-40,
            edge_count_delta=-110,
            community_count_delta=-3,
            god_node_jaccard_similarity=None,
            god_nodes_entered=None,
            god_nodes_exited=None,
        ),
    )
    dumped = event.model_dump(mode="json")
    restored = CodebaseDeltaV1.model_validate(dumped)
    assert restored == event
    assert restored.graph.god_node_jaccard_similarity is None


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        CodebaseDeltaV1.model_validate(
            {"domain": "git", "observed_at": _NOW.isoformat(), "unexpected_field": 1}
        )


def test_domain_field_mismatch_rejected_when_declared_field_missing() -> None:
    """domain='git' but the git field itself is None -- must fail validation,
    not silently construct a schema-valid-but-empty-shell event."""
    with pytest.raises(ValidationError, match="git"):
        CodebaseDeltaV1(domain="git", observed_at=_NOW)


def test_domain_field_mismatch_rejected_when_wrong_field_populated() -> None:
    """domain='git' but pr_lifecycle is populated instead -- must fail, not
    silently accept a payload whose declared domain doesn't match its real
    content."""
    with pytest.raises(ValidationError, match="git"):
        CodebaseDeltaV1(
            domain="git",
            observed_at=_NOW,
            pr_lifecycle=PrLifecycleDeltaPayloadV1(
                since=_NOW, until=_NOW, submitted_count=1, merged_count=0, closed_without_merge_count=0
            ),
        )


def test_domain_field_mismatch_rejected_when_extra_domain_also_populated() -> None:
    """domain='git' with a real git payload AND a real graph payload both set
    -- the design spec's contract is exactly one domain per event, not a
    convenient multi-domain batch; must be rejected, not silently accepted
    with the extra field ignored."""
    with pytest.raises(ValidationError, match="git"):
        CodebaseDeltaV1(
            domain="git",
            observed_at=_NOW,
            git=GitDeltaPayloadV1(
                prev_sha="a" * 40, head_sha="b" * 40, commit_count=1,
                files_added=0, files_deleted=0, files_modified=1,
                lines_added=10, lines_removed=0,
            ),
            graph=GraphDeltaPayloadV1(node_count_delta=1, edge_count_delta=1, community_count_delta=0),
        )


def test_domain_must_be_one_of_the_three_literals() -> None:
    with pytest.raises(ValidationError):
        CodebaseDeltaV1.model_validate({"domain": "not_a_real_domain", "observed_at": _NOW.isoformat()})
