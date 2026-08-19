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
from orion.schemas.proposal_frame import ProposalFrameV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _proposal() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick:test",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame:test",
        overall_action_pressure=0.4,
        overall_risk=0.1,
        candidates=[],
    )


def test_worker_skips_when_no_proposal_pending(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = PolicyRuntimeWorker()
    worker._store.load_next_proposal_without_policy_frame = MagicMock(return_value=None)
    worker._store.save_policy_decision_frame = MagicMock()
    # _tick now calls these, and they hit the database on a store built from a dummy URI.
    worker._store.reconcile_policy_pending = MagicMock(return_value=0)
    worker._store.load_policy_frame_for_proposal = MagicMock(return_value=None)
    worker._store.clear_policy_pending = MagicMock(return_value=1)

    worker._tick()

    worker._store.save_policy_decision_frame.assert_not_called()


def test_worker_saves_policy_frame_for_pending_proposal(monkeypatch) -> None:
    # 2026-07-22 (SelfStateV1 burn): build_policy_decision_frame now
    # evaluates directly off proposal_frame, so a pending proposal always
    # produces a saved decision frame -- no separate self-state load that
    # could fail and stall the FIFO queue.
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = PolicyRuntimeWorker()
    proposal = _proposal()
    worker._store.load_next_proposal_without_policy_frame = MagicMock(return_value=proposal)
    worker._store.save_policy_decision_frame = MagicMock()
    worker._store.reconcile_policy_pending = MagicMock(return_value=0)
    worker._store.load_policy_frame_for_proposal = MagicMock(return_value=None)
    worker._store.clear_policy_pending = MagicMock(return_value=1)

    worker._tick()

    worker._store.save_policy_decision_frame.assert_called_once()
    saved_frame = worker._store.save_policy_decision_frame.call_args[0][0]
    assert saved_frame.source_proposal_frame_id == proposal.frame_id
    assert saved_frame.source_field_tick_id == proposal.source_field_tick_id


def test_worker_does_not_redecide_an_already_decided_proposal(monkeypatch) -> None:
    """The marker defaults to TRUE, so a stale-true row reaches _tick with a decision already
    in place. Re-deciding is NOT a harmless repeat: stable_policy_frame_id() is deterministic
    and the insert is ON CONFLICT DO UPDATE, so it would overwrite the original decision with a
    fresh evaluation under today's policy and today's timestamp -- rewriting history silently.
    The old anti-join made that structurally impossible; the marker does not."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = PolicyRuntimeWorker()
    proposal = _proposal()
    worker._store.load_next_proposal_without_policy_frame = MagicMock(return_value=proposal)
    worker._store.save_policy_decision_frame = MagicMock()
    worker._store.reconcile_policy_pending = MagicMock(return_value=0)
    worker._store.load_policy_frame_for_proposal = MagicMock(return_value=object())
    worker._store.clear_policy_pending = MagicMock(return_value=1)

    worker._tick()

    worker._store.save_policy_decision_frame.assert_not_called()
    worker._store.clear_policy_pending.assert_called_once_with(proposal.frame_id)
