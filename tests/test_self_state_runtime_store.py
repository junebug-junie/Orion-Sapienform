from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-self-state-runtime"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_store_class():
    spec = importlib.util.spec_from_file_location(
        "self_state_runtime_store",
        SVC / "app" / "store.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SelfStateRuntimeStore


SelfStateRuntimeStore = _load_store_class()

from orion.schemas.self_state import SelfStateV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:tick_a:frame_a:self_state_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_a",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame_a",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.6,
    )


def test_save_and_load_latest(monkeypatch) -> None:
    payload = _state().model_dump(mode="json")
    store = SelfStateRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_self_state" in sql:
            result.rowcount = 1
        elif "source_attention_frame_id" in sql:
            result.mappings.return_value.first.return_value = None
        else:
            result.mappings.return_value.first.return_value = {"self_state_json": payload}
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_self_state(_state())
    loaded = store.load_latest_self_state()
    assert loaded is not None
    assert loaded.self_state_id == "self.state:tick_a:frame_a:self_state_policy.v1"


def test_load_by_attention_frame_id(monkeypatch) -> None:
    payload = _state().model_dump(mode="json")
    store = SelfStateRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "source_attention_frame_id" in str(stmt):
            result.mappings.return_value.first.return_value = {"self_state_json": payload}
        else:
            result.mappings.return_value.first.return_value = None
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    loaded = store.load_self_state_for_attention_frame("frame_a")
    assert loaded is not None
    assert loaded.source_attention_frame_id == "frame_a"
