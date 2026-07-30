from __future__ import annotations

from types import SimpleNamespace

from app.chat_stance import _project_autonomy_from_beliefs


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
    """Graph snapshot_source=drive_state must not populate drive_state or autonomy summary."""
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
    # drive_state snapshots are ignored entirely; dominant_drive falls back to
    # first DriveNode label when no autonomy snapshot is present.
    assert result["summary"].dominant_drive == "coherence"
    assert result["summary"].top_drives == ["coherence", "continuity"]
    assert result["summary"].raw_state_present is False


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


# The Postgres drive_audits projection tests that used to live here
# (test_build_chat_stance_inputs_uses_postgres_drive_state and siblings,
# exercising app/drive_state_postgres.py's fetch_drive_state_for_chat_stance
# via ctx["chat_drive_state"]/CHAT_STANCE_DRIVE_STATE_VISIBLE) were removed
# 2026-07-30 (chore/delete-orion-drives Wave 2a) along with that module:
# drive_audits' sole producer was already deleted in Wave 1, so this was
# testing a read against a frozen, producer-less table, not live behavior.
# The tests above this comment (_project_autonomy_from_beliefs / substrate
# DriveNode + snapshot_source="autonomy" path) are unrelated and still real.
