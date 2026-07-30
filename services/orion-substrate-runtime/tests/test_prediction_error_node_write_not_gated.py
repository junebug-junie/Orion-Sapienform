"""Structural gate: no `_write_prediction_error_node()` call may sit inside an
`if error > 0.0:` block.

This is deliberately an AST check over `worker.py`'s source rather than five
separate tick fixtures. The bug it guards has now been found three separate
times on three different domains (`bus_synaptic` frozen at a stale 1.0 for
hours, 2026-07-30; `route` frozen at a stale 0.00025 while its tick kept
running; `chat` likewise), and each time it was fixed for exactly the one
domain that happened to be noticed. The failure is structural and identical
every time, so the guard should be structural too -- CLAUDE.md §4: "when a
latent failure repeats, turn it into deterministic code, a test, an eval, or
a skill."

Why gating the node write is wrong, in one line: a domain whose
`prediction_error` is legitimately 0.0 on a quiet tick keeps its last
NON-ZERO value in the graph forever, so every consumer that polls the node's
current value (`orion-equilibrium-service`'s transport gate, AST/HOT's
`prediction_error_by_domain`) reads a stale high-water mark as if it were a
present-tense reading -- a signal that can rise but can never come back down
to a genuine calm state. That is precisely the failure mode CLAUDE.md's
metric quality gate step 4 exists to catch.

The receipt write (`save_receipt`) staying inside the gate is correct and is
NOT flagged here: receipts are an audit trail of notable prediction-error
events, not a polled current-state read.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKER_PATH = Path(__file__).resolve().parents[1] / "app" / "worker.py"


def _is_error_gt_zero_test(node: ast.expr) -> bool:
    """Match `error > 0.0` (and `error > 0`)."""
    if not isinstance(node, ast.Compare):
        return False
    if not (isinstance(node.left, ast.Name) and node.left.id == "error"):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Gt):
        return False
    comparator = node.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == 0


def _node_write_calls(tree: ast.AST) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "_write_prediction_error_node":
            out.append(node)
    return out


def test_no_prediction_error_node_write_is_gated_on_error_greater_than_zero() -> None:
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))

    gated: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _is_error_gt_zero_test(node.test):
            continue
        for stmt in node.body:
            for call in _node_write_calls(stmt):
                # Recover the node_id kwarg so a failure names the domain
                # rather than just a line number.
                node_id = "<unknown>"
                for kw in call.keywords:
                    if kw.arg == "node_id" and isinstance(kw.value, ast.Constant):
                        node_id = str(kw.value.value)
                gated.append((call.lineno, node_id))

    assert not gated, (
        "_write_prediction_error_node() must be called on EVERY tick, not only when "
        "error > 0.0 -- a gated write leaves the node holding its last non-zero value "
        "indefinitely on a quiet domain, which every polling consumer reads as a current "
        "reading. Move the call out of the `if error > 0.0:` block (the save_receipt call "
        "stays gated). Offending call sites: "
        + ", ".join(f"{node_id} at worker.py:{lineno}" for lineno, node_id in gated)
    )


def test_every_prediction_error_domain_still_writes_its_node() -> None:
    """Companion to the above: moving a call out of the gate must not have
    silently deleted it. Asserts all five Active-Inference domains plus
    transport still have a node write somewhere in the module."""
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))

    written: set[str] = set()
    for call in _node_write_calls(tree):
        for kw in call.keywords:
            if kw.arg == "node_id" and isinstance(kw.value, ast.Constant):
                written.add(str(kw.value.value))

    for domain in ("biometrics", "execution", "chat", "route", "bus_synaptic"):
        assert f"node:substrate.{domain}" in written, (
            f"node:substrate.{domain} has no _write_prediction_error_node() call at all"
        )
