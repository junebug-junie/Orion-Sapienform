"""Phase G — compaction applier (memory mutation, hard-gated).

Every safety invariant is exercised against an in-memory fake store: default-off,
policy-gated (operator_review can't self-approve), snapshot-precedes-apply,
downscale-first / prune-gated, rollback-on-error, never-raises.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.compaction import (
    ConsolidateEntryV1,
    DownscaleEntryV1,
    MemoryCompactionDeltaV1,
    PruneEntryV1,
)
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1

NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)


class FakeStore:
    """Records call order; models a rollback-able before/after state."""

    def __init__(self, *, fail_on: str | None = None, fail_snapshot: bool = False):
        self.calls: list[str] = []
        self.weights = {"e-1": 0.9, "e-2": 0.8}
        self.episodics = {"ep-1", "ep-2"}
        self.cards: list[str] = []
        self._fail_on = fail_on
        self._fail_snapshot = fail_snapshot

    def snapshot(self, delta):
        self.calls.append("snapshot")
        if self._fail_snapshot:
            raise RuntimeError("snapshot boom")
        return {"weights": dict(self.weights), "episodics": set(self.episodics), "cards": list(self.cards)}

    def restore(self, snapshot):
        self.calls.append("restore")
        self.weights = dict(snapshot["weights"])
        self.episodics = set(snapshot["episodics"])
        self.cards = list(snapshot["cards"])

    def downscale(self, target_id, old_w, new_w):
        self.calls.append(f"downscale:{target_id}")
        if self._fail_on == "downscale":
            raise RuntimeError("downscale boom")
        self.weights[target_id] = new_w

    def prune(self, episodic_id):
        self.calls.append(f"prune:{episodic_id}")
        if self._fail_on == "prune":
            raise RuntimeError("prune boom")
        self.episodics.discard(episodic_id)

    def write_gist(self, gist_card, evidence_refs, supersedes):
        self.calls.append("write_gist")
        if self._fail_on == "write_gist":
            raise RuntimeError("gist boom")
        self.cards.append(gist_card)


def _delta(delta_id="d-1"):
    return MemoryCompactionDeltaV1(
        delta_id=delta_id,
        consolidate=[ConsolidateEntryV1(gist_card="a settled theme", evidence_refs=["ol-1"])],
        downscale=[DownscaleEntryV1(target_id="e-1", old_w=0.9, new_w=0.4, reason="settled")],
        prune=[PruneEntryV1(episodic_id="ep-1", salience=0.05, ttl_reason="stale")],
    )


def _approval(delta_id="d-1", *, execution_allowed=True, decision="approved_for_execution",
              policy_gate="execution_policy"):
    return PolicyDecisionFrameV1(
        frame_id="pf-1",
        generated_at=NOW,
        source_proposal_frame_id="prop-frame-1",
        source_self_state_id="ss-1",
        overall_risk=0.2,
        execution_allowed=execution_allowed,
        approved_decisions=[
            PolicyDecisionV1(
                decision_id="dec-1",
                proposal_id=delta_id,
                decision=decision,
                policy_gate=policy_gate,
                risk_score=0.2,
                reversibility_score=1.0,
                confidence_score=0.8,
            )
        ],
    )


# --- gating -------------------------------------------------------------------

def test_disabled_gate_touches_nothing(tmp_path, monkeypatch):
    from app import compaction_applier as ca

    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=False)
    assert r.status == "disabled"
    assert r.applied is False
    assert store.calls == []  # not even a snapshot


def test_not_approved_when_operator_review_only():
    from app import compaction_applier as ca

    store = FakeStore()
    # An operator_review proposal that was NOT execution-approved.
    frame = _approval(execution_allowed=False, decision="requires_operator_review",
                      policy_gate="operator_review")
    r = ca.apply_compaction_delta(_delta(), frame, store=store, enabled=True)
    assert r.status == "not_approved"
    assert store.calls == []


def test_not_approved_when_frame_missing():
    from app import compaction_applier as ca

    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), None, store=store, enabled=True)
    assert r.status == "not_approved"
    assert store.calls == []


def test_not_approved_when_approval_references_other_delta():
    from app import compaction_applier as ca

    store = FakeStore()
    r = ca.apply_compaction_delta(_delta("d-1"), _approval("d-OTHER"), store=store, enabled=True)
    assert r.status == "not_approved"


def test_isolated_decision_guard_rejects_operator_review_decision():
    """execution_allowed=True (frame approved for *something*) but this decision
    is still requires_operator_review → must fail closed. Isolates the
    decision-level guard so deleting it would fail this test."""
    from app import compaction_applier as ca

    frame = _approval(execution_allowed=True, decision="requires_operator_review",
                      policy_gate="operator_review")
    r = ca.apply_compaction_delta(_delta(), frame, store=FakeStore(), enabled=True)
    assert r.status == "not_approved"


def test_isolated_gate_guard_rejects_non_execution_gate():
    """approved_for_execution but gated by operator_review (not execution_policy),
    frame execution_allowed=True → must fail closed. Isolates the gate guard."""
    from app import compaction_applier as ca

    frame = _approval(execution_allowed=True, decision="approved_for_execution",
                      policy_gate="operator_review")
    r = ca.apply_compaction_delta(_delta(), frame, store=FakeStore(), enabled=True)
    assert r.status == "not_approved"


def test_autonomy_policy_gate_is_not_accepted_for_memory_mutation():
    """The autonomy engine's self-authorized gate must NOT authorize this rung —
    a human is required. Fails closed even with approved_for_execution."""
    from app import compaction_applier as ca

    frame = _approval(execution_allowed=True, decision="approved_for_execution",
                      policy_gate="autonomy_policy")
    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), frame, store=store, enabled=True)
    assert r.status == "not_approved"
    assert store.calls == []  # not even a snapshot


def test_approval_does_not_leak_across_deltas_sharing_a_request():
    """A human approving delta A must not authorize un-reviewed delta B that
    merely shares a source_request_id. Regression for the request-fallback leak."""
    from app import compaction_applier as ca

    delta_a = MemoryCompactionDeltaV1(
        delta_id="dA",
        source_request_ids=["r1"],
        downscale=[DownscaleEntryV1(target_id="e-1", old_w=0.9, new_w=0.4)],
    )
    delta_b = MemoryCompactionDeltaV1(
        delta_id="dB",
        source_request_ids=["r1"],  # same request, different (unseen) op set
        prune=[PruneEntryV1(episodic_id="ep-1", salience=0.01)],
    )
    # Frame approves ONLY delta A (proposal_id + evidence ref name A / r1).
    frame = PolicyDecisionFrameV1(
        frame_id="pf", generated_at=NOW, source_proposal_frame_id="pfr",
        source_self_state_id="ss", overall_risk=0.2, execution_allowed=True,
        approved_decisions=[
            PolicyDecisionV1(
                decision_id="d", proposal_id="dA", decision="approved_for_execution",
                policy_gate="execution_policy", risk_score=0.2, reversibility_score=1.0,
                confidence_score=0.8, evidence_refs=["r1"],
            )
        ],
    )
    assert ca.policy_approves_execution(delta_a, frame) is True
    assert ca.policy_approves_execution(delta_b, frame) is False  # no leak


def test_malformed_frame_fails_closed():
    """A duck-typed/malformed frame must fail closed, not raise."""
    from app import compaction_applier as ca

    class Bad:
        execution_allowed = True
        approved_decisions = None  # would AttributeError on iteration

    assert ca.policy_approves_execution(_delta(), Bad()) is False


def test_empty_delta_is_empty(monkeypatch):
    from app import compaction_applier as ca

    r = ca.apply_compaction_delta(
        MemoryCompactionDeltaV1(delta_id="d-empty"),
        _approval("d-empty"),
        store=FakeStore(),
        enabled=True,
    )
    assert r.status == "empty"


# --- happy path + ordering ----------------------------------------------------

def test_snapshot_precedes_every_write(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True, downscale_only=True)
    assert r.status == "applied" and r.applied is True
    # snapshot is the very first call, before any mutation.
    assert store.calls[0] == "snapshot"
    assert "restore" not in store.calls
    # a snapshot artifact was written (§14).
    assert r.snapshot_path is not None
    assert (tmp_path / "d-1" / "snapshot_before.json").exists()


def test_downscale_only_skips_prune(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True, downscale_only=True)
    assert r.downscaled == 1
    assert r.pruned == 0
    assert r.prune_skipped_downscale_only == 1
    assert r.cards_written == 1
    assert not any(c.startswith("prune") for c in store.calls)  # prune never ran
    assert store.weights["e-1"] == 0.4  # downscale applied
    assert store.episodics == {"ep-1", "ep-2"}  # nothing pruned


def test_prune_runs_when_downscale_only_false(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True, downscale_only=False)
    assert r.pruned == 1
    assert "ep-1" not in store.episodics
    # downscale still ran strictly before prune.
    assert store.calls.index("downscale:e-1") < store.calls.index("prune:ep-1")


# --- rollback -----------------------------------------------------------------

def test_rollback_restores_on_write_error(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore(fail_on="write_gist")
    before_weights = dict(store.weights)
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True, downscale_only=True)
    assert r.status == "rolled_back"
    assert r.applied is False
    assert "restore" in store.calls
    # the downscale that ran before the failure was reverted by the restore.
    assert store.weights == before_weights


def test_snapshot_failure_fails_closed(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore(fail_snapshot=True)
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True)
    assert r.status == "rolled_back"
    # no mutation was attempted — only the (failed) snapshot.
    assert store.calls == ["snapshot"]


def test_never_raises_even_if_rollback_also_fails(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))

    class DoubleFail(FakeStore):
        def restore(self, snapshot):
            self.calls.append("restore")
            raise RuntimeError("restore boom")

    store = DoubleFail(fail_on="downscale")
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True)
    assert r.status == "rolled_back"
    assert r.error and "rollback_failed" in r.error


def test_module_has_no_default_canonical_store():
    """Phase G must not bind a real canonical-memory store — importing it touches
    nothing. The store is always injected."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "compaction_applier.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    # No sqlalchemy / psycopg import — no path to a live DB from this module.
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    for mod in mods:
        assert "sqlalchemy" not in mod.lower(), "applier must not open a DB itself"
        assert "psycopg" not in mod.lower(), "applier must not open a DB itself"
