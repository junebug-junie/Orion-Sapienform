from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-policy-runtime"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_store_class():
    spec = importlib.util.spec_from_file_location(
        "policy_runtime_store",
        SVC / "app" / "store.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PolicyRuntimeStore


PolicyRuntimeStore = _load_store_class()

from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _frame() -> PolicyDecisionFrameV1:
    decision = PolicyDecisionV1(
        decision_id="policy.decision:p1:substrate_policy.v1",
        proposal_id="p1",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
    )
    return PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:test:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:test:proposal_policy.v1",
        source_field_tick_id="field.tick:test",
        decisions=[decision],
        approved_decisions=[decision],
        overall_risk=0.05,
    )


def test_save_and_load_latest(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = PolicyRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_policy_decision_frames" in sql:
            result.rowcount = 1
        elif "source_proposal_frame_id" in sql and "ORDER BY" in sql:
            result.mappings.return_value.first.return_value = None
        else:
            result.mappings.return_value.first.return_value = {
                "policy_decision_frame_json": payload,
            }
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_policy_decision_frame(_frame())
    loaded = store.load_latest_policy_decision_frame()
    assert loaded is not None
    assert loaded.frame_id == _frame().frame_id


def test_load_by_proposal_frame_id(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = PolicyRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "source_proposal_frame_id" in str(stmt):
            result.mappings.return_value.first.return_value = {
                "policy_decision_frame_json": payload,
            }
        else:
            result.mappings.return_value.first.return_value = None
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    loaded = store.load_policy_frame_for_proposal("proposal.frame:test:proposal_policy.v1")
    assert loaded is not None
    assert loaded.source_proposal_frame_id == "proposal.frame:test:proposal_policy.v1"


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


def test_load_next_proposal_without_policy_frame_retires_incompatible_row(monkeypatch) -> None:
    # Live incident (2026-07-22): the SelfStateV1 burn removed
    # source_self_state_id from ProposalFrameV1. This is the FIFO "oldest
    # proposal without a policy frame" lookup -- a naive raise here
    # crash-loops the whole worker forever (confirmed live), since this
    # exact row is always "the oldest unresolved" until a policy frame
    # exists for it. It must degrade to None AND write a stub decision
    # frame so the FIFO advances past the bad row.
    store = PolicyRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    insert_calls: list[dict] = []

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_policy_decision_frames" in sql:
            insert_calls.append(params or {})
            result.rowcount = 1
        else:
            result.mappings.return_value.first.return_value = {
                "proposal_frame_json": _legacy_self_state_proposal_payload(),
            }
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_next_proposal_without_policy_frame()

    assert result is None
    assert len(insert_calls) == 1
    assert insert_calls[0]["source_proposal_frame_id"] == "proposal.frame:legacy:proposal_policy.v1"


def test_save_idempotent_by_frame_id(monkeypatch) -> None:
    store = PolicyRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value = MagicMock(rowcount=1)
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_policy_decision_frame(_frame())
    sql = str(conn.execute.call_args[0][0])
    assert "ON CONFLICT (frame_id)" in sql
