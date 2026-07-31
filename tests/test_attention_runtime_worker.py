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
from orion.attention.field_attention.candidate_precision_weighted import (  # noqa: E402
    NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
    NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
    PrecisionEwmaBaseline,
)
from orion.attention.field_attention.selectors import (  # noqa: E402
    PREDICTION_ERROR_NATIVE_TARGETS,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1  # noqa: E402
from orion.schemas.field_state import FieldStateV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_existing",
        node_vectors={"node:athena": {"cortex_exec_step_load": 1.0}},
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


def test_worker_tick_advances_baseline_per_native_target_and_wires_into_frame(monkeypatch) -> None:
    """2026-07-30 EWMA-baseline fix regression test (code review finding: the
    dict-comprehension in _tick() that calls advance_node_prediction_error_baseline
    per PREDICTION_ERROR_NATIVE_TARGETS entry, and wires the result into
    build_attention_frame via prediction_error_baselines=, had no test coverage --
    only the pieces (store method, selector, builder) were tested in isolation).

    Confirms: (1) advance_node_prediction_error_baseline is called exactly once per
    real native target, with the real reducer_key mapping and the real EWMA
    alpha/min_variance constants and the configured fetch_limit; (2) the returned
    baselines actually reach the saved frame (a real node target appears, scored
    off the mocked baseline), not silently dropped somewhere in the wiring.
    """
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = AttentionRuntimeWorker()
    worker._store.load_latest_field = MagicMock(return_value=_field())
    worker._store.load_attention_frame_for_field_tick = MagicMock(return_value=None)
    worker._store.load_latest_attention_frame = MagicMock(return_value=None)
    worker._store.save_attention_frame = MagicMock()

    real_baseline = PrecisionEwmaBaseline(
        ewma=0.1, variance=0.02, observation_count=25, last_value=0.3
    )
    advance_mock = MagicMock(return_value=real_baseline)
    worker._store.advance_node_prediction_error_baseline = advance_mock

    worker._tick()

    assert advance_mock.call_count == len(PREDICTION_ERROR_NATIVE_TARGETS)
    called_target_ids = set()
    for call in advance_mock.call_args_list:
        kwargs = call.kwargs
        called_target_ids.add(kwargs["target_id"])
        assert kwargs["reducer_key"] == PREDICTION_ERROR_NATIVE_TARGETS[kwargs["target_id"]]
        assert kwargs["alpha"] == NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA
        assert kwargs["min_variance"] == NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        assert kwargs["fetch_limit"] == worker._settings.prediction_error_history_limit
    assert called_target_ids == set(PREDICTION_ERROR_NATIVE_TARGETS)

    worker._store.save_attention_frame.assert_called_once()
    saved_frame = worker._store.save_attention_frame.call_args[0][0]
    saved_target_ids = {t.target_id for t in saved_frame.node_targets}
    # Every real native target got the same mocked baseline (observation_count=25,
    # i.e. fully qualified) -- all five should appear as real node targets, not be
    # silently excluded, confirming the baselines dict genuinely reached the builder.
    assert saved_target_ids == set(PREDICTION_ERROR_NATIVE_TARGETS)
