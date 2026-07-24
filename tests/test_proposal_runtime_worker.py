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

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_existing",
        node_vectors={"node:athena": {"cortex_exec_step_load": 1.0}},
    )


def _existing_frame() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:tick_existing:frame_existing:proposal_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_existing",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_existing:field_attention_policy.v1",
        overall_action_pressure=0.4,
        overall_risk=0.1,
    )


def test_worker_skips_when_proposal_exists_for_field_tick(monkeypatch) -> None:
    """2026-07-22 (SelfStateV1 burn): the poll trigger is FieldStateV1 directly
    now -- worker skips a tick already covered by an existing frame, keyed on
    field_tick_id instead of a self-state ID."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ProposalRuntimeWorker()
    worker._store.load_latest_field = MagicMock(return_value=_field())
    worker._store.load_proposal_frame_for_field_tick = MagicMock(return_value=_existing_frame())
    worker._store.save_proposal_frame = MagicMock()

    worker._tick()

    worker._store.save_proposal_frame.assert_not_called()


def test_worker_skips_when_field_missing(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ProposalRuntimeWorker()
    worker._store.load_latest_field = MagicMock(return_value=None)
    worker._store.save_proposal_frame = MagicMock()

    worker._tick()

    worker._store.save_proposal_frame.assert_not_called()
