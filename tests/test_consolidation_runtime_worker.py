from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-consolidation-runtime"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SVC) not in sys.path:
    sys.path.insert(0, str(SVC))


def _load_worker_class():
    spec = importlib.util.spec_from_file_location(
        "consolidation_runtime_worker",
        SVC / "app" / "worker.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConsolidationRuntimeWorker


ConsolidationRuntimeWorker = _load_worker_class()

from orion.consolidation.windows import compute_consolidation_window, stable_consolidation_frame_id  # noqa: E402
from orion.schemas.consolidation_frame import ConsolidationFrameV1  # noqa: E402

NOW = datetime(2026, 5, 25, 15, 37, tzinfo=timezone.utc)


def test_tick_skips_when_frame_exists_for_bucket(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ConsolidationRuntimeWorker()
    window_start, window_end = compute_consolidation_window(now=NOW, lookback_minutes=60)
    frame_id = stable_consolidation_frame_id(
        window_start=window_start,
        window_end=window_end,
        policy_id=worker._policy.policy_id,
    )
    existing = ConsolidationFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        window_start=window_start,
        window_end=window_end,
    )

    worker._store.load_consolidation_frame_for_window = MagicMock(return_value=existing)
    worker._store.load_window_data = MagicMock()
    worker._store.save_consolidation_frame = MagicMock()

    with patch(
        "orion.consolidation.windows.compute_consolidation_window",
        return_value=(window_start, window_end),
    ):
        worker._tick()

    worker._store.load_window_data.assert_not_called()
    worker._store.save_consolidation_frame.assert_not_called()
