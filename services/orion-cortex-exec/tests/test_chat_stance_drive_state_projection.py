from __future__ import annotations

from types import SimpleNamespace

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


# --- unit tests on _project_autonomy_from_beliefs directly (SimpleNamespace fixtures,
# matching the pattern in test_chat_stance_self_state_projection.py) ---


def test_drive_state_snapshot_source_is_now_collected_and_populates_drive_state_key():
    """Gap 2 fix: snapshot_source == 'drive_state' must be picked up, not silently dropped."""
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
    ds = result["drive_state"]
    assert ds is not None
    assert ds["pressures"] == {"coherence": 0.8, "continuity": 0.6}
    assert ds["activations"] == {"coherence": True, "continuity": False}
    assert ds["dominant_drive"] == "coherence"
    assert ds["summary"] == "orion pressure concentrates on coherence"
    assert ds["tension_kinds"] == ["drive_competition.coherence_continuity"]


def test_drive_state_fields_taken_atomically_from_one_snapshot_not_mixed_across_several():
    """Regression guard for the cross-snapshot field-mixing bug: if a snapshot
    with only dominant_drive sits ahead of a later snapshot with only
    activations/summary/tension_kinds, fields must NOT be mixed across them --
    the first content-bearing snapshot wins entirely, or nothing from it does."""
    beliefs = SimpleNamespace(
        anchors={
            "orion": _anchor_slice(
                snapshots=[
                    _snapshot_node("drive_state", {"dominant_drive": "coherence"}),
                    _snapshot_node(
                        "drive_state",
                        {
                            "activations": {"continuity": True},
                            "summary": "continuity rising",
                            "tension_kinds": ["unresolved_thread"],
                        },
                    ),
                ],
            )
        }
    )
    result = _project_autonomy_from_beliefs(beliefs, {})
    assert result is not None
    ds = result["drive_state"]
    assert ds is not None
    # First snapshot (dominant_drive only) wins entirely -- nothing from the
    # second snapshot bleeds in, even though it has real content too.
    assert ds["dominant_drive"] == "coherence"
    assert ds["activations"] == {}
    assert ds["summary"] is None
    assert ds["tension_kinds"] == []


def test_autonomy_snapshot_source_still_works_alongside_drive_state():
    """The legacy 'autonomy' snapshot_source producer (autonomy_ctx.py, gated behind
    AUTONOMY_GRAPH_BACKEND=graphdb/sparql) must keep working -- it is a real,
    registered producer, not dead code, so the filter clause is kept."""
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
    # No drive_state-sourced snapshot present -> no activations/summary/dominant_drive
    # from the drive_state.v1 lineage, even though pressures still reflect the shared
    # drive nodes (drive nodes aren't source-scoped).
    ds = result["drive_state"]
    assert ds is not None
    assert ds["pressures"] == {"relational": 0.5}
    assert ds["activations"] == {}
    assert ds["dominant_drive"] is None
    assert ds["summary"] is None


def test_no_drive_or_snapshot_nodes_returns_none():
    beliefs = SimpleNamespace(anchors={"orion": _anchor_slice()})
    assert _project_autonomy_from_beliefs(beliefs, {}) is None


def test_none_beliefs_returns_none():
    assert _project_autonomy_from_beliefs(None, {}) is None


# --- integration tests through build_chat_stance_inputs: verify inputs["drive_state"]
# is a sibling of inputs["autonomy"], gated behind CHAT_STANCE_DRIVE_STATE_VISIBLE,
# and never merged into inputs["autonomy"]. ---


def _real_beliefs_with_drive_state() -> UnifiedRelationalBeliefSetV1:
    anchor_slice = AnchorBeliefSliceV1(
        anchor="orion",
        drives=[_drive_node("coherence", 0.8)],
        snapshots=[
            _snapshot_node(
                "drive_state",
                {
                    "activations": {"coherence": True},
                    "artifact_id": "drive-state-1",
                    "dominant_drive": "coherence",
                    "summary": "orion pressure concentrates on coherence",
                },
            )
        ],
    )
    return UnifiedRelationalBeliefSetV1(anchors={"orion": anchor_slice})


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_exposes_drive_state_as_sibling_when_flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _real_beliefs_with_drive_state())
    monkeypatch.setenv("CHAT_STANCE_DRIVE_STATE_VISIBLE", "true")

    ctx = {"user_message": "hello"}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    assert "drive_state" in built
    assert built["drive_state"]["pressures"] == {"coherence": 0.8}
    assert built["drive_state"]["dominant_drive"] == "coherence"
    assert built["drive_state"]["summary"] == "orion pressure concentrates on coherence"

    # `autonomy` and `drive_state` are structurally independent siblings -- `drive_state`
    # is a top-level `inputs` key, never nested inside or blended into `autonomy`'s dict.
    assert "autonomy" in built
    assert built["autonomy"].keys() >= {"state", "summary", "debug"}
    assert "pressures" not in built["autonomy"]
    assert "activations" not in built["autonomy"]
    assert "drive_state" not in built["autonomy"]
    # autonomy's own `summary` is the AutonomySummaryV1 dump, not drive_state's summary string.
    assert built["autonomy"]["summary"] != built["drive_state"]["summary"]


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_omits_drive_state_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _real_beliefs_with_drive_state())
    monkeypatch.delenv("CHAT_STANCE_DRIVE_STATE_VISIBLE", raising=False)

    ctx = {"user_message": "hello"}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    assert "drive_state" not in built
    assert "autonomy" in built
