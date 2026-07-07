"""Phase B — spontaneous thought → governed proposal candidate.

The security-critical guarantees: a reverie proposal always carries a policy
gate (never auto-dispatch), degrades to None on weak/hollow thoughts, and the
reverie path never imports orion-actions (awake proposes, substrate disposes).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.reverie.proposal import (
    REVERIE_PROPOSAL_SOURCE,
    spontaneous_thought_to_candidate,
)
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

GROUNDED = "The deploy loop ol-1 keeps winning and has not discharged; conflict at n-1 recurs."


def _coalition():
    return CoalitionSnapshotV1(
        attended_node_ids=["n-1"],
        selected_open_loop_id="ol-1",
        open_loop_ids=["ol-1"],
        generated_at="2026-07-06T00:00:00Z",
    )


def _thought(salience=0.8, interpretation=GROUNDED, evidence=("ol-1",), coalition=True):
    return SpontaneousThoughtV1(
        thought_id="th-1", correlation_id="c-1",
        coalition=_coalition() if coalition else None,
        interpretation=interpretation, salience=salience, evidence_refs=list(evidence),
    ).marked_hollow()


def test_grounded_thought_becomes_review_gated_candidate():
    c = spontaneous_thought_to_candidate(_thought(), self_state_id="ss-1")
    assert c is not None
    assert c.source == REVERIE_PROPOSAL_SOURCE
    assert c.thought_id == "th-1"
    assert c.proposal_kind == "request_policy_review"


def test_candidate_always_requires_policy_gate_never_auto():
    # The load-bearing safety invariant: a spontaneous proposal can NEVER carry a
    # gate that would let it auto-dispatch.
    c = spontaneous_thought_to_candidate(_thought(salience=1.0), self_state_id="ss-1")
    assert c is not None
    assert c.required_policy_gate == "operator_review"
    assert c.required_policy_gate not in ("none", "read_only")
    assert c.proposed_effect == "prepare_for_policy_gate"


def test_autoaction_posture_recorded_but_gate_unchanged():
    # Arming ORION_REVERIE_AUTOACTION_ENABLED records an inspectable posture but
    # must NOT lower the gate — the load-bearing Phase B safety property.
    off = spontaneous_thought_to_candidate(_thought(), self_state_id="ss-1")
    on = spontaneous_thought_to_candidate(_thought(), self_state_id="ss-1", autoaction_enabled=True)
    assert "reverie_autoaction:off" in off.reasons
    assert "reverie_autoaction:on" in on.reasons
    # The gate is operator_review in BOTH cases — auto-action never auto-dispatches.
    assert off.required_policy_gate == "operator_review"
    assert on.required_policy_gate == "operator_review"


def test_hollow_thought_yields_no_candidate():
    hollow = _thought(evidence=(), interpretation="x")  # short + no evidence
    assert spontaneous_thought_to_candidate(hollow, self_state_id="ss-1") is None


def test_low_salience_yields_no_candidate():
    assert spontaneous_thought_to_candidate(_thought(salience=0.2), self_state_id="ss-1") is None


def test_absent_coalition_targets_self_state_without_raising():
    # A thought with no coalition is hollow → None, never a raise.
    assert spontaneous_thought_to_candidate(_thought(coalition=False), self_state_id="ss-1") is None


def test_evidence_refs_capped():
    t = _thought(evidence=["ol-1"])  # grounded
    c = spontaneous_thought_to_candidate(t, self_state_id="ss-1")
    assert len(c.evidence_refs) <= 200


def test_semantic_lift_audit_ref_thought_survives_proposal_path():
    # A semantic-lift thought cites an audit ref (a loop source_ref) that is NOT
    # in strict grounding_ids(). It was stamped non-hollow via extra_grounding.
    # The proposal path must trust that stamp, not recompute strictly and drop it.
    t = SpontaneousThoughtV1(
        thought_id="th-2", correlation_id="c-2",
        coalition=_coalition(),
        interpretation=GROUNDED,
        salience=0.8,
        evidence_refs=["harness_closure:corr-2"],  # outside strict grounding_ids()
    ).marked_hollow(extra_grounding={"harness_closure:corr-2"})
    assert not t.hollow
    # Sanity: without the stamp's extra_grounding, a strict recompute would drop it.
    assert t.hollow_reason_for() == "unanchored_evidence_outside_coalition"
    c = spontaneous_thought_to_candidate(t, self_state_id="ss-1")
    assert c is not None
    assert c.thought_id == "th-2"


# --- contract: the reverie proposal path must never touch orion-actions --------

def test_reverie_proposal_never_imports_orion_actions():
    # Scan only real import statements (not docstrings/comments) via the AST.
    import ast

    src = Path(__file__).resolve().parents[1] / "proposal.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for mod in imported:
        assert "action" not in mod.lower(), f"reverie proposal must not import {mod!r}"
