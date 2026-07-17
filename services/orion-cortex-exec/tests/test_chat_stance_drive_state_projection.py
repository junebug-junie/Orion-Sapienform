from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app import chat_stance
from app.chat_stance import _project_autonomy_from_beliefs
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1


def _drive_node(drive_kind: str, salience: float) -> SimpleNamespace:
    return SimpleNamespace(
        node_kind="drive",
        drive_kind=drive_kind,
        signals=SimpleNamespace(salience=salience),
        metadata={},
    )


def _snapshot_node(snapshot_source: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(node_kind="state_snapshot", snapshot_source=snapshot_source, metadata=metadata)


def _anchor_slice(*, drives=None, snapshots=None, goals=None, tensions=None, degraded=False) -> SimpleNamespace:
    return SimpleNamespace(
        drives=drives or [],
        goals=goals or [],
        tensions=tensions or [],
        snapshots=snapshots or [],
        degraded=degraded,
        tier_outcomes=[],
    )


def test_graph_drive_state_snapshot_is_not_sor_for_drive_state_key():
    """Graph snapshot_source=drive_state must not populate drive_state (Postgres is SoR)."""
    beliefs = SimpleNamespace(
        anchors={
            "orion": _anchor_slice(
                drives=[_drive_node("coherence", 0.8), _drive_node("continuity", 0.6)],
                snapshots=[
                    _snapshot_node(
                        "drive_state",
                        {
                            "activations": {"coherence": True, "continuity": False},
                            "artifact_id": "drive-state-1",
                            "dominant_drive": "coherence",
                            "summary": "orion pressure concentrates on coherence",
                            "tension_kinds": ["drive_competition.coherence_continuity"],
                        },
                    )
                ],
            )
        }
    )
    result = _project_autonomy_from_beliefs(beliefs, {})
    assert result is not None
    assert result["drive_state"] is None
    # Autonomy summary may still see dominant_drive from the snapshot metadata
    # for legacy autonomy-lineage fields; measurement SoR is Postgres-only.
    assert result["summary"].dominant_drive == "coherence"


def test_autonomy_snapshot_source_still_works_without_drive_state_projection():
    beliefs = SimpleNamespace(
        anchors={
            "orion": _anchor_slice(
                drives=[_drive_node("relational", 0.5)],
                snapshots=[
                    _snapshot_node(
                        "autonomy",
                        {"dominant_drive": "relational", "identity_summary": "steady"},
                    )
                ],
            )
        }
    )
    result = _project_autonomy_from_beliefs(beliefs, {})
    assert result is not None
    assert result["summary"].dominant_drive == "relational"
    assert result["drive_state"] is None


def test_no_drive_or_snapshot_nodes_returns_none():
    beliefs = SimpleNamespace(anchors={"orion": _anchor_slice()})
    assert _project_autonomy_from_beliefs(beliefs, {}) is None


def test_none_beliefs_returns_none():
    assert _project_autonomy_from_beliefs(None, {}) is None


def _real_beliefs_without_drive_state() -> UnifiedRelationalBeliefSetV1:
    anchor_slice = AnchorBeliefSliceV1(
        anchor="orion",
        drives=[_drive_node("coherence", 0.8)],
        snapshots=[
            _snapshot_node(
                "autonomy",
                {"dominant_drive": "coherence"},
            )
        ],
    )
    return UnifiedRelationalBeliefSetV1(anchors={"orion": anchor_slice})


async def _fake_postgres_drive_state(correlation_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "pressures": {"coherence": 0.8},
            "activations": {"coherence": True},
            "dominant_drive": "coherence",
            "summary": "orion pressure concentrates on coherence",
            "tension_kinds": ["drive_competition.coherence_continuity"],
        },
        {"ok": True, "reason": "success", "source": "drive_audits", "correlation_id": correlation_id},
    )


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_uses_postgres_drive_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _real_beliefs_without_drive_state())
    monkeypatch.setattr(
        "app.drive_state_postgres.fetch_drive_state_for_chat_stance",
        _fake_postgres_drive_state,
    )
    monkeypatch.setenv("CHAT_STANCE_DRIVE_STATE_VISIBLE", "true")

    ctx = {"user_message": "hello", "correlation_id": "corr-stance-1"}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    assert ctx["chat_drive_state"]["dominant_drive"] == "coherence"
    assert ctx["chat_drive_state"]["pressures"] == {"coherence": 0.8}
    assert ctx["chat_drive_state"]["tension_kinds"] == ["drive_competition.coherence_continuity"]
    assert ctx["chat_drive_state_diagnostics"]["source"] == "drive_audits"
    assert "drive_state" in built
    assert built["drive_state"]["summary"] == "orion pressure concentrates on coherence"
    assert "autonomy" in built
    assert built["autonomy"].keys() >= {"state", "summary", "debug"}
    assert "pressures" not in built["autonomy"]
    assert "drive_state" not in built["autonomy"]


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_omits_prompt_drive_state_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _real_beliefs_without_drive_state())
    monkeypatch.setattr(
        "app.drive_state_postgres.fetch_drive_state_for_chat_stance",
        _fake_postgres_drive_state,
    )
    monkeypatch.delenv("CHAT_STANCE_DRIVE_STATE_VISIBLE", raising=False)

    ctx = {"user_message": "hello"}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    assert "drive_state" not in built
    assert "autonomy" in built
    assert ctx["chat_drive_state"]["dominant_drive"] == "coherence"


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_fail_open_when_postgres_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _empty(_correlation_id: str) -> tuple[None, dict[str, Any]]:
        return None, {"ok": True, "reason": "no_rows", "source": "drive_audits"}

    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _real_beliefs_without_drive_state())
    monkeypatch.setattr("app.drive_state_postgres.fetch_drive_state_for_chat_stance", _empty)
    monkeypatch.setenv("CHAT_STANCE_DRIVE_STATE_VISIBLE", "true")

    ctx = {"user_message": "hello"}
    built = await chat_stance.build_chat_stance_inputs(ctx)
    assert "chat_drive_state" not in ctx
    assert "drive_state" not in built
    assert ctx["chat_drive_state_diagnostics"]["reason"] == "no_rows"
