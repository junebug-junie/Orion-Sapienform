from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from stale_channel_node_cleanup import compute_stale_channel_candidates  # type: ignore


def test_literal_uuid_reply_channel_is_a_candidate() -> None:
    catalog_names = {"orion:vision:reply:*"}
    rows = [
        {"channel": "orion:vision:reply:7486bd55-b055-44ce-981e-ae5b95d8032d", "edge_count": 1},
    ]

    candidates = compute_stale_channel_candidates(rows, catalog_names)

    assert len(candidates) == 1
    assert candidates[0]["normalizes_to"] == "orion:vision:reply:*"


def test_real_catalog_entry_is_never_a_candidate() -> None:
    catalog_names = {"orion:system:health"}
    rows = [{"channel": "orion:system:health", "edge_count": 500000}]

    candidates = compute_stale_channel_candidates(rows, catalog_names)

    assert candidates == []


def test_wildcard_node_itself_is_never_a_candidate() -> None:
    # The wildcard target node's own name normalizes to itself -- it must
    # never be selected for deletion, only the literal nodes that collapse
    # *into* it.
    catalog_names = {"orion:vision:reply:*"}
    rows = [{"channel": "orion:vision:reply:*", "edge_count": 1501}]

    candidates = compute_stale_channel_candidates(rows, catalog_names)

    assert candidates == []


def test_unmatched_literal_channel_with_no_wildcard_is_not_a_candidate() -> None:
    # A channel with no catalog wildcard pattern at all -- normalize_channel_name
    # returns it unchanged, so it must not be treated as stale debt.
    catalog_names = {"orion:system:health"}
    rows = [{"channel": "orion:some:undeclared:channel", "edge_count": 1}]

    candidates = compute_stale_channel_candidates(rows, catalog_names)

    assert candidates == []


def test_mixed_batch_only_flags_the_stale_ones() -> None:
    catalog_names = {"orion:system:health", "orion:vision:reply:*"}
    rows = [
        {"channel": "orion:system:health", "edge_count": 500000},
        {"channel": "orion:vision:reply:*", "edge_count": 1501},
        {"channel": "orion:vision:reply:aaaa-bbbb", "edge_count": 1},
        {"channel": "orion:vision:reply:cccc-dddd", "edge_count": 1},
    ]

    candidates = compute_stale_channel_candidates(rows, catalog_names)

    assert {c["channel"] for c in candidates} == {
        "orion:vision:reply:aaaa-bbbb",
        "orion:vision:reply:cccc-dddd",
    }
