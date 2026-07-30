"""Unit tests for ``orion/structural_mass/snapshot_history.py``."""

from __future__ import annotations

import json
from pathlib import Path

from orion.structural_mass.snapshot_history import (
    GraphSnapshotStats,
    append_snapshot,
    graph_snapshot_stats_from_text,
    parse_graph_report_god_nodes,
    read_snapshots,
)

_REAL_GOD_NODES_SECTION = """\
Some preamble text.

## God Nodes (most connected - your core abstractions)
1. `ServiceRef` - 1033 edges
2. `BaseEnvelope` - 980 edges
3. `OrionBusAsync` - 628 edges

## Surprising Connections (you probably didn't know these)
- `A` --relates_to--> `B`
"""


def test_parse_graph_report_god_nodes_extracts_ordered_labels() -> None:
    labels = parse_graph_report_god_nodes(_REAL_GOD_NODES_SECTION)
    assert labels == ("ServiceRef", "BaseEnvelope", "OrionBusAsync")


def test_parse_graph_report_god_nodes_missing_section_is_none() -> None:
    """No `## God Nodes` header at all -- distinct from a header present with
    a genuinely empty body. Returns None (unknown), not () (known-empty)."""
    assert parse_graph_report_god_nodes("# Report\n\nNo god nodes section here.\n") is None


def test_parse_graph_report_god_nodes_format_change_is_none_not_empty() -> None:
    """Regression guard for the 2026-07-30 code-review finding: if graphify's
    report format ever changes (e.g. labels no longer backtick-wrapped, a
    bullet list instead of a numbered one), the header is present with real
    content but zero lines match the expected pattern -- must report None
    (parse failure), not silently collapse to () ("genuinely zero god
    nodes"), which would make graph_structural_delta() misreport a broken
    parse as "nothing changed" (god_node_jaccard_similarity=1.0)."""
    text = (
        "## God Nodes (most connected - your core abstractions)\n"
        "1. ServiceRef - 1033 edges\n"  # no backticks -- doesn't match the real format
        "2. BaseEnvelope - 980 edges\n"
    )
    assert parse_graph_report_god_nodes(text) is None


def test_parse_graph_report_god_nodes_empty_section_is_empty() -> None:
    text = "## God Nodes (most connected - your core abstractions)\n\n## Next Section\nsomething\n"
    assert parse_graph_report_god_nodes(text) == ()


def test_parse_graph_report_god_nodes_stops_at_next_header() -> None:
    """A god node label containing text that looks like a numbered list item
    in a *later* section must not be picked up -- the section boundary is the
    next `## ` heading, not end-of-file."""
    text = (
        "## God Nodes (most connected - your core abstractions)\n"
        "1. `Real` - 10 edges\n"
        "\n"
        "## Other Section\n"
        "1. `NotAGodNode` - irrelevant\n"
    )
    assert parse_graph_report_god_nodes(text) == ("Real",)


def test_graph_snapshot_stats_from_text_computes_counts() -> None:
    graph_json = json.dumps(
        {
            "nodes": [
                {"id": "a", "community": 1},
                {"id": "b", "community": 1},
                {"id": "c", "community": 2},
            ],
            "links": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ],
        }
    )
    stats = graph_snapshot_stats_from_text(
        graph_json, _REAL_GOD_NODES_SECTION, commit_sha="abc123", backfilled=True
    )
    assert stats.node_count == 3
    assert stats.edge_count == 2
    assert stats.community_count == 2
    assert stats.god_nodes == ("ServiceRef", "BaseEnvelope", "OrionBusAsync")
    assert stats.commit_sha == "abc123"
    assert stats.backfilled is True


def test_graph_snapshot_stats_from_text_handles_missing_community_field() -> None:
    """A node with no `community` key at all (schema-legal, just incomplete)
    must not raise or silently count as its own community bucket."""
    graph_json = json.dumps(
        {
            "nodes": [{"id": "a"}, {"id": "b", "community": 5}],
            "links": [],
        }
    )
    stats = graph_snapshot_stats_from_text(graph_json, "")
    assert stats.node_count == 2
    assert stats.community_count == 1


def test_snapshot_json_round_trip_preserves_none_god_nodes() -> None:
    """None (unparseable) must survive a JSON round-trip as None, not silently
    become () -- json.dumps(None) is `null`, and from_json_dict must read
    that back as None, not coerce it to an empty tuple."""
    original = GraphSnapshotStats(node_count=5, edge_count=3, community_count=1, god_nodes=None)
    restored = GraphSnapshotStats.from_json_dict(original.to_json_dict())
    assert restored.god_nodes is None
    assert restored == original


def test_snapshot_json_round_trip() -> None:
    original = GraphSnapshotStats(
        node_count=10,
        edge_count=20,
        community_count=3,
        god_nodes=("X", "Y"),
        commit_sha="deadbeef",
        backfilled=True,
    )
    restored = GraphSnapshotStats.from_json_dict(original.to_json_dict())
    assert restored == original


def test_append_and_read_snapshots_round_trip(tmp_path: Path) -> None:
    log_path = tmp_path / "history" / "graph_snapshots.jsonl"
    s1 = GraphSnapshotStats(node_count=1, edge_count=0, community_count=1, god_nodes=())
    s2 = GraphSnapshotStats(node_count=2, edge_count=1, community_count=1, god_nodes=("X",))

    append_snapshot(log_path, s1)
    append_snapshot(log_path, s2)

    snapshots = read_snapshots(log_path)
    assert snapshots == [s1, s2]


def test_read_snapshots_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert read_snapshots(tmp_path / "does_not_exist.jsonl") == []


def test_append_snapshot_creates_parent_directories(tmp_path: Path) -> None:
    log_path = tmp_path / "deeply" / "nested" / "path" / "snapshots.jsonl"
    append_snapshot(log_path, GraphSnapshotStats(node_count=1, edge_count=0, community_count=1))
    assert log_path.exists()
