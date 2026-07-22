from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-proposal-runtime"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_store_class():
    spec = importlib.util.spec_from_file_location(
        "proposal_runtime_store",
        SVC / "app" / "store.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ProposalRuntimeStore


ProposalRuntimeStore = _load_store_class()

from orion.schemas.proposal_frame import ProposalFrameV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _frame() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:tick:frame:proposal_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_a",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame_a",
        overall_action_pressure=0.5,
        overall_risk=0.1,
    )


def test_save_and_load_latest(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = ProposalRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_proposal_frames" in sql:
            result.rowcount = 1
        elif "source_field_tick_id" in sql and "ORDER BY" in sql:
            result.mappings.return_value.first.return_value = None
        else:
            result.mappings.return_value.first.return_value = {"proposal_frame_json": payload}
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_proposal_frame(_frame())
    loaded = store.load_latest_proposal_frame()
    assert loaded is not None
    assert loaded.frame_id == _frame().frame_id


def _legacy_self_state_proposal_payload() -> dict:
    # Shaped like a pre-2026-07-22 (SelfStateV1 burn) proposal frame row --
    # source_self_state_id/source_self_state_generated_at no longer exist on
    # ProposalFrameV1, and source_field_generated_at is now required.
    return {
        "schema_version": "proposal.frame.v1",
        "frame_id": "proposal.frame:legacy:proposal_policy.v1",
        "generated_at": NOW.isoformat(),
        "source_self_state_id": "self.state:legacy",
        "source_self_state_generated_at": NOW.isoformat(),
        "source_attention_frame_id": "attention.frame:legacy",
        "source_field_tick_id": "tick:legacy",
        "overall_action_pressure": 0.5,
        "overall_risk": 0.1,
        "candidates": [],
    }


def test_load_latest_proposal_frame_degrades_to_none_on_legacy_row(monkeypatch) -> None:
    # Live incident (2026-07-22): the SelfStateV1 burn removed
    # source_self_state_id from ProposalFrameV1 and added a required
    # source_field_generated_at. A pre-migration row is the single latest
    # row until the poll loop successfully writes a new one -- a naive
    # raise here crash-loops the whole worker forever (confirmed live).
    store = ProposalRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "proposal_frame_json": _legacy_self_state_proposal_payload(),
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.load_latest_proposal_frame() is None


def test_load_by_field_tick_id_degrades_to_none_on_legacy_row(monkeypatch) -> None:
    store = ProposalRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "proposal_frame_json": _legacy_self_state_proposal_payload(),
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.load_proposal_frame_for_field_tick("tick:legacy") is None


def test_load_by_field_tick_id(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = ProposalRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "source_field_tick_id" in str(stmt):
            result.mappings.return_value.first.return_value = {"proposal_frame_json": payload}
        else:
            result.mappings.return_value.first.return_value = None
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    loaded = store.load_proposal_frame_for_field_tick("tick_a")
    assert loaded is not None
    assert loaded.source_field_tick_id == "tick_a"


def test_save_idempotent_by_frame_id(monkeypatch) -> None:
    store = ProposalRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    calls: list[str] = []

    def execute_side_effect(stmt, params=None):
        calls.append(str(stmt))
        return MagicMock()

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_proposal_frame(_frame())
    assert any("ON CONFLICT (frame_id)" in sql for sql in calls)
