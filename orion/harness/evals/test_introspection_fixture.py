"""Gate tests pinning the introspection fixture to the actual repository.

Every evidence symbol named by the fixture must literally exist in its file —
this is the deterministic drift check that replaces a breadcrumb parser in v1.
"""

from __future__ import annotations

from pathlib import Path

from orion.harness.evals.introspection_scoring import (
    extract_tool_metrics,
    load_fixture,
    score_answer,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_fixture_loads_with_unique_question_ids() -> None:
    fixture = load_fixture()
    ids = [q["id"] for q in fixture["questions"]]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 9


def test_every_question_assertion_is_defined() -> None:
    fixture = load_fixture()
    defined = set(fixture["assertions"])
    for question in fixture["questions"]:
        for aid in question["assertions"]:
            assert aid in defined, f"{question['id']} references undefined assertion {aid}"


def test_every_assertion_has_patterns() -> None:
    fixture = load_fixture()
    for aid, spec in fixture["assertions"].items():
        assert spec.get("any_of"), f"assertion {aid} has no any_of patterns"


def test_evidence_symbols_exist_in_named_files() -> None:
    fixture = load_fixture()
    for aid, spec in fixture["assertions"].items():
        evidence = spec.get("evidence")
        if not evidence:
            continue
        path = REPO_ROOT / evidence["file"]
        assert path.is_file(), f"assertion {aid}: missing evidence file {evidence['file']}"
        text = path.read_text(encoding="utf-8")
        assert evidence["symbol"] in text, (
            f"assertion {aid}: symbol {evidence['symbol']!r} no longer in {evidence['file']}"
        )


def test_score_answer_passes_and_fails() -> None:
    fixture = load_fixture()
    answer = (
        "The turn enters through websocket_endpoint, which gates on "
        "ORION_UNIFIED_TURN_ENABLED and calls run_unified_turn; "
        "execute_unified_turn owns the Hub-side saga."
    )
    passed, failed = score_answer(
        answer, ["route_gate", "hub_orchestration_boundary", "draft_not_final"], fixture["assertions"]
    )
    assert passed == ["route_gate", "hub_orchestration_boundary"]
    assert failed == ["draft_not_final"]


def test_extract_tool_metrics_counts_tools_and_files() -> None:
    frames = [
        {
            "type": "step",
            "step": {
                "type": "assistant",
                "context_fill_pct": 12.5,
                "raw": {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "name": "Read", "input": {"file_path": "/repo/a.py"}},
                            {"type": "tool_use", "name": "mcp__gitnexus__query", "input": {"search_query": "unified turn"}},
                        ]
                    },
                },
            },
        },
        {
            "type": "step",
            "step": {
                "type": "assistant",
                "context_fill_pct": 31.0,
                "raw": {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "name": "Read", "input": {"file_path": "/repo/a.py"}}
                        ]
                    },
                },
            },
        },
    ]
    metrics = extract_tool_metrics(frames)
    assert metrics.tool_calls == 3
    assert metrics.tool_names.count("Read") == 2
    assert metrics.unique_files_read == 1
    assert metrics.max_context_fill_pct == 31.0
    assert metrics.observed_chars > 0
