from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-proposal-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from app.worker import ProposalRuntimeWorker  # noqa: E402
from orion.schemas.field_state import FieldStateV1  # noqa: E402
from orion.schemas.proposal_frame import ProposalFrameV1  # noqa: E402
from orion.schemas.self_state import SelfStateV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:tick_existing:frame_existing:self_state_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_existing",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_existing:field_attention_policy.v1",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.7,
    )


def _field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_existing",
        node_vectors={"node:athena": {"execution_load": 1.0}},
    )


def _existing_frame() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:self.state:tick_existing:frame_existing:self_state_policy.v1:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id="self.state:tick_existing:frame_existing:self_state_policy.v1",
        source_self_state_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_existing:field_attention_policy.v1",
        source_field_tick_id="tick_existing",
        overall_action_pressure=0.4,
        overall_risk=0.1,
    )


def test_worker_skips_when_proposal_exists_for_self_state(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ProposalRuntimeWorker()
    worker._store.load_latest_self_state = MagicMock(return_value=_self_state())
    worker._store.load_proposal_frame_for_self_state = MagicMock(return_value=_existing_frame())
    worker._store.save_proposal_frame = MagicMock()

    worker._tick()

    worker._store.save_proposal_frame.assert_not_called()


def test_worker_skips_when_field_missing(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ProposalRuntimeWorker()
    worker._store.load_latest_self_state = MagicMock(return_value=_self_state())
    worker._store.load_proposal_frame_for_self_state = MagicMock(return_value=None)
    worker._store.load_attention_frame = MagicMock(return_value=None)
    worker._store.load_field_for_tick = MagicMock(return_value=None)
    worker._store.save_proposal_frame = MagicMock()

    worker._tick()

    worker._store.save_proposal_frame.assert_not_called()
