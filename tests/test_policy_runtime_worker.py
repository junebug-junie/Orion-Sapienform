from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-policy-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from app.worker import PolicyRuntimeWorker  # noqa: E402
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1  # noqa: E402
from orion.schemas.proposal_frame import ProposalFrameV1  # noqa: E402
from orion.schemas.self_state import SelfStateV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:test",
        generated_at=NOW,
        source_field_tick_id="tick:test",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame:test",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.9,
    )


def _proposal() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id="self.state:test",
        source_self_state_generated_at=NOW,
        source_attention_frame_id="frame:test",
        source_field_tick_id="tick:test",
        overall_action_pressure=0.4,
        overall_risk=0.1,
        candidates=[],
    )


def _existing_policy_frame() -> PolicyDecisionFrameV1:
    return PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:test:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:test:proposal_policy.v1",
        source_self_state_id="self.state:test",
        overall_risk=0.0,
    )


def test_worker_skips_when_no_proposal_pending(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = PolicyRuntimeWorker()
    worker._store.load_next_proposal_without_policy_frame = MagicMock(return_value=None)
    worker._store.save_policy_decision_frame = MagicMock()

    worker._tick()

    worker._store.save_policy_decision_frame.assert_not_called()


def test_worker_records_unevaluable_frame_when_self_state_missing(monkeypatch) -> None:
    # 2026-07-12: a naive skip-and-return here would retry the exact same
    # oldest unresolved proposal forever (live incident -- a schema change
    # made an old self-state row permanently unloadable, and this proposal
    # is always "the oldest without a policy frame" until one exists for
    # it, blocking every proposal queued behind it). The worker must still
    # record a frame -- an honest "could not evaluate" one -- so the FIFO
    # queue advances past it.
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = PolicyRuntimeWorker()
    proposal = _proposal()
    worker._store.load_next_proposal_without_policy_frame = MagicMock(return_value=proposal)
    worker._store.load_self_state = MagicMock(return_value=None)
    worker._store.save_policy_decision_frame = MagicMock()

    worker._tick()

    worker._store.save_policy_decision_frame.assert_called_once()
    saved_frame = worker._store.save_policy_decision_frame.call_args[0][0]
    assert saved_frame.source_proposal_frame_id == proposal.frame_id
    assert saved_frame.decisions == []
    assert saved_frame.operator_review_required is True
    assert saved_frame.execution_allowed is False
    assert any("self_state" in w for w in saved_frame.warnings)


def test_worker_saves_policy_frame_for_pending_proposal(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = PolicyRuntimeWorker()
    proposal = _proposal()
    worker._store.load_next_proposal_without_policy_frame = MagicMock(return_value=proposal)
    worker._store.load_self_state = MagicMock(return_value=_self_state())
    worker._store.save_policy_decision_frame = MagicMock()

    worker._tick()

    worker._store.save_policy_decision_frame.assert_called_once()
