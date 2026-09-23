from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker


def _worker(monkeypatch, *, enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")
    monkeypatch.setenv(
        "SUBSTRATE_SYSTEM_ONE_APPRAISAL_ENABLED",
        "true" if enabled else "false",
    )
    monkeypatch.setenv("SUBSTRATE_SYSTEM_ONE_BASE_URL", "http://kev:8009")

    import app.settings as settings_mod

    settings_mod._settings = None
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._substrate_graph_store = None
    worker._pending_system_one_grammar_events = []
    return worker


def _graph_node(node_id: str, label: str, pressure: float) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        label=label,
        metadata={"dynamic_pressure": pressure},
        signals=SimpleNamespace(confidence=0.8),
    )


def test_system_one_failure_does_not_break_attention_broadcast(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={
            "node:hot": _graph_node(
                "node:hot", "unresolved contradiction", 0.9
            )
        }
    )
    worker._system_one_appraisal_tick = MagicMock(
        side_effect=RuntimeError("kev unavailable")
    )

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()  # must not raise

    worker._store.save_attention_broadcast.assert_called_once()
    worker._system_one_appraisal_tick.assert_called_once()


def test_disabled_system_one_is_not_called(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=False)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={
            "node:hot": _graph_node(
                "node:hot", "unresolved contradiction", 0.9
            )
        }
    )
    worker._system_one_appraisal_tick = MagicMock()

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()

    worker._store.save_attention_broadcast.assert_called_once()
    worker._system_one_appraisal_tick.assert_not_called()


def test_system_one_tick_omits_stale_field_frame(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=True)
    stale = MagicMock()
    stale.generated_at = __import__("datetime").datetime(
        2020, 1, 1, tzinfo=__import__("datetime").timezone.utc
    )
    worker._store.get_latest_field_attention_frame.return_value = stale
    worker._store.save_system_one_appraisal = MagicMock()

    frame = MagicMock()
    frame.model_id = "kev-latest"
    frame.frame_id = "frame-1"
    frame.latency_ms = 5
    frame.answers = {}

    with patch(
        "orion.substrate.system_one_appraisal.run_system_one_appraisal",
        return_value=frame,
    ) as run, patch(
        "orion.substrate.system_one_appraisal.build_system_one_grammar_events",
        return_value=[],
    ):
        worker._system_one_appraisal_tick(broadcast=MagicMock())

    assert run.call_args.kwargs["field_frame"] is None
    worker._store.save_system_one_appraisal.assert_called_once()


def test_system_one_tick_does_not_persist_when_inference_fails(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=True)
    worker._store.get_latest_field_attention_frame.return_value = None

    with patch(
        "orion.substrate.system_one_appraisal.run_system_one_appraisal",
        side_effect=ValueError("incomplete provider response"),
    ):
        try:
            worker._system_one_appraisal_tick(broadcast=MagicMock())
        except ValueError:
            pass
        else:
            raise AssertionError("expected inference failure")

    worker._store.save_system_one_appraisal.assert_not_called()
