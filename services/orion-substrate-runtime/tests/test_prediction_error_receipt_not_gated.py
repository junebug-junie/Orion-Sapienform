"""Every prediction-error receipt must be saved on every tick, including a calm
tick whose value is exactly 0.0.

The receipt is the only path into (a) orion-field-digester's field node vector
`prediction_error` channel and (b) orion-attention-runtime's Candidate A
precision baseline (`substrate_node_prediction_error_baseline`, whose
`last_value` is used as the target's current error). Gating it on
`error > 0.0` froze both at the last non-zero value. Confirmed live 2026-09-25:
node:substrate.route's field value sat at 0.0003 with node_vector_updated_at
12h+ old while its tick kept running and writing 0.0 to the FalkorDB node.

Sibling of test_prediction_error_node_write_not_gated.py, which covered only
the FalkorDB node write; that earlier fix explicitly left the receipt gated,
which is how this bug survived it.
"""

from __future__ import annotations

import ast
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

import app.worker as worker_module
from app.worker import BiometricsSubstrateWorker
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1

WORKER_PATH = SUBSTRATE_ROOT / "app" / "worker.py"
_NOW = datetime(2026, 9, 25, 0, 0, 0, tzinfo=timezone.utc)

# Every node that has a _prediction_error_receipt() call site today.
_RECEIPT_NODE_IDS = {
    "node:substrate.biometrics",
    "node:substrate.execution",
    "node:substrate.chat",
    "node:substrate.route",
    "node:substrate.bus_synaptic",
    "node:substrate.codebase",
    "node:substrate.vision",
    "node:substrate.perception",
}


def _is_gt_zero(test: ast.expr) -> bool:
    """Any `<expr> > 0` / `<expr> > 0.0` comparison (error, result.score, ...)."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Gt):
        return False
    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == 0


def _receipt_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prediction_error_receipt"
    ]


def _node_id(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "node_id" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return "<unknown>"


def test_no_prediction_error_receipt_is_gated_on_value_greater_than_zero() -> None:
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
    gated: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _is_gt_zero(node.test):
            continue
        for stmt in node.body:
            for call in _receipt_calls(stmt):
                gated.append((call.lineno, _node_id(call)))
    assert not gated, (
        "_prediction_error_receipt() must be saved on EVERY tick, including 0.0 -- it is "
        "the only write path into the field's prediction_error channel and the attention "
        "precision baseline, so a gated receipt freezes both at the last non-zero value. "
        "Offending call sites: "
        + ", ".join(f"{node_id} at worker.py:{lineno}" for lineno, node_id in gated)
    )


def test_every_domain_still_has_a_receipt_call() -> None:
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
    found = {_node_id(call) for call in _receipt_calls(tree)}
    assert _RECEIPT_NODE_IDS <= found, _RECEIPT_NODE_IDS - found


def _grammar_event(event_id: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id="trace-1",
        emitted_at=_NOW,
        provenance=GrammarProvenanceV1(source_service="orion-hub"),
    )


def test_route_tick_calm_error_saves_zero_receipt(monkeypatch) -> None:
    """The exact live incident: route_prediction_error() returns 0.0 on a quiet
    tick. The field-bound receipt must still be saved, carrying 0.0."""
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._store = MagicMock()
    worker._store.fetch_route_grammar_events.return_value = [_grammar_event("gev-1")]
    worker._write_prediction_error_node = MagicMock()

    monkeypatch.setattr(worker_module, "process_route_grammar_events", lambda **kwargs: None)
    monkeypatch.setattr(worker_module, "route_prediction_error", lambda prev, curr: 0.0)

    worker._route_tick()

    worker._store.save_receipt.assert_called_once()
    receipt = worker._store.save_receipt.call_args[0][0]
    delta = receipt.state_deltas[0]
    assert delta.target_kind == "prediction_signal"
    assert delta.target_id == "node:substrate.route"
    assert delta.after["pressure_hints"]["prediction_error"] == 0.0
    # Graph node still written too (the earlier, graph-only half of this fix).
    worker._write_prediction_error_node.assert_called_once()
    assert worker._write_prediction_error_node.call_args.kwargs["error"] == 0.0

