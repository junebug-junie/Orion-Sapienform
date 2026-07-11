"""Gate tests pinning the introspection fixture to the actual repository.

Every evidence symbol named by the fixture must literally exist in its file —
this is the deterministic drift check that replaces a breadcrumb parser in v1.
"""

from __future__ import annotations

import ast
from pathlib import Path

from orion.harness.evals.introspection_scoring import (
    extract_tool_metrics,
    load_fixture,
    score_answer,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _strip_docstrings(text: str) -> str:
    """Remove module/class/function docstrings so evidence anchors must live in
    code, not in breadcrumb prose that would keep the gate green after a rename."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                text = text.replace(doc, "", 1)
    return text


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


def test_no_needle_is_echo_passable_from_its_question_prompt() -> None:
    """An answer that merely restates the question must score zero: no any_of
    needle may appear verbatim in the prompt of a question that uses it."""
    fixture = load_fixture()
    assertions = fixture["assertions"]
    for question in fixture["questions"]:
        prompt = question["prompt"].lower()
        for aid in question["assertions"]:
            for needle in assertions[aid]["any_of"]:
                assert needle.lower() not in prompt, (
                    f"{question['id']}: needle {needle!r} of assertion {aid} is "
                    "echo-passable from the question prompt"
                )


def test_evidence_symbols_exist_in_named_files() -> None:
    fixture = load_fixture()
    for aid, spec in fixture["assertions"].items():
        evidence = spec.get("evidence")
        if not evidence:
            continue
        path = REPO_ROOT / evidence["file"]
        assert path.is_file(), f"assertion {aid}: missing evidence file {evidence['file']}"
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            text = _strip_docstrings(text)
        assert evidence["symbol"] in text, (
            f"assertion {aid}: symbol {evidence['symbol']!r} no longer in the code "
            f"of {evidence['file']} (docstrings don't count)"
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
    """Frame shapes mirror the runtime: annotate_harness_step puts fill under
    step['context_obs']['fill_pct'] (orion/fcc/context_budget.py)."""
    frames = [
        {
            "type": "step",
            "step": {
                "type": "assistant",
                "context_obs": {"fill_pct": 12},
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
                "context_obs": {"fill_pct": 31},
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
    assert metrics.truncated_results == 0


def test_extract_tool_metrics_counts_truncations_in_tool_results_only() -> None:
    marker = "[orion-fcc-mcp-proxy: truncated 90000 chars to 12000. ...]"
    frames = [
        {
            "type": "step",
            "step": {
                "type": "user",
                "raw": {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "content": [
                                    {"type": "text", "text": f"a {marker}"},
                                    {"type": "text", "text": f"b {marker}"},
                                ],
                            }
                        ]
                    },
                },
            },
        },
        # The marker as plain assistant text (e.g. the motor read the proxy
        # module's source) must NOT count as a truncation.
        {
            "type": "step",
            "step": {
                "type": "assistant",
                "raw": {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": marker}]},
                },
            },
        },
    ]
    metrics = extract_tool_metrics(frames)
    assert metrics.truncated_results == 2
