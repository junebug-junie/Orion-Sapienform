from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-attention-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from app.worker import AttentionRuntimeWorker  # noqa: E402
from orion.schemas.field_attention_frame import FieldAttentionFrameV1  # noqa: E402
from orion.schemas.field_state import FieldStateV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_existing",
        node_vectors={"node:athena": {"execution_load": 1.0}},
    )


def _existing_frame() -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="attention.frame:tick_existing:field_attention_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_existing",
        source_field_generated_at=NOW,
        overall_salience=0.5,
    )


def test_worker_skips_when_frame_exists_for_tick(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = AttentionRuntimeWorker()
    worker._store.load_latest_field = MagicMock(return_value=_field())
    worker._store.load_attention_frame_for_field_tick = MagicMock(return_value=_existing_frame())
    worker._store.save_attention_frame = MagicMock()

    worker._tick()

    worker._store.save_attention_frame.assert_not_called()
