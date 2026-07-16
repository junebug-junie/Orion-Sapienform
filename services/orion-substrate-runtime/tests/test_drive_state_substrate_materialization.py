"""Unit tests for materializing DriveEngine's drive_state/drive_audit into the
substrate graph (snapshot_source="drive_state"), so chat_stance.py's
drive-state projection actually receives real data instead of reading an
always-empty snapshot list -- see
orion/autonomy/drives_and_autonomy_retrospective.md §9.

Mirrors test_embodiment_c_hook.py's worker-construction pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker

from orion.core.schemas.drives import ArtifactProvenance, DriveAuditV1, DriveStateV1


def _make_worker(monkeypatch, *, materialization_enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED",
        "true" if materialization_enabled else "false",
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._bus = MagicMock()
    worker._latest_drive_state = None
    worker._latest_drive_state_at = None
    worker._latest_drive_audit = None
    worker._latest_drive_audit_at = None
    worker._substrate_graph_store = None
    return worker


def _drive(pressures: dict[str, float]) -> DriveStateV1:
    return DriveStateV1(
        subject="orion",
        model_layer="drive",
        entity_id="orion",
        kind="memory.drives.state.v1",
        provenance=ArtifactProvenance(intake_channel="test"),
        pressures=pressures,
        activations={k: v >= 0.5 for k, v in pressures.items()},
    )


def _audit(**overrides) -> DriveAuditV1:
    base = dict(
        subject="orion",
        model_layer="drive",
        entity_id="orion",
        kind="memory.drives.audit.v1",
        provenance=ArtifactProvenance(intake_channel="test"),
        active_drives=["coherence"],
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.72},
        tension_kinds=["drive_competition.coherence_continuity"],
        summary="coherence pressure elevated",
    )
    base.update(overrides)
    return DriveAuditV1(**base)


def test_materialization_disabled_writes_nothing(monkeypatch):
    worker = _make_worker(monkeypatch, materialization_enabled=False)
    store = MagicMock()
    worker._get_substrate_graph_store = MagicMock(return_value=store)
    worker._latest_drive_state = _drive({"coherence": 0.72})

    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(
        ok=True, envelope=MagicMock(payload=_drive({"coherence": 0.72}).model_dump(mode="json"))
    )
    worker._bus = bus
    worker._cache_drive_state_message({"data": b"payload"})

    store.upsert_node.assert_not_called()


def test_materialization_enabled_upserts_state_snapshot_and_drive_nodes(monkeypatch):
    worker = _make_worker(monkeypatch, materialization_enabled=True)
    store = MagicMock()
    worker._get_substrate_graph_store = MagicMock(return_value=store)
    worker._latest_drive_audit = _audit()

    drive = _drive({"coherence": 0.72, "continuity": 0.1})
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=MagicMock(payload=drive.model_dump(mode="json")))
    worker._bus = bus
    worker._cache_drive_state_message({"data": b"payload"})

    assert worker._latest_drive_state is not None
    assert store.upsert_node.call_count >= 1
    node_kinds = {call.kwargs["node"].__class__.__name__ for call in store.upsert_node.call_args_list}
    assert "StateSnapshotNodeV1" in node_kinds

    # The state snapshot node must carry drive_state="drive_state" and the
    # audit's tension_kinds through its metadata -- this is the exact field
    # chat_stance.py's drive-state projection reads.
    snapshot_calls = [
        c for c in store.upsert_node.call_args_list if c.kwargs["node"].__class__.__name__ == "StateSnapshotNodeV1"
    ]
    assert len(snapshot_calls) == 1
    node = snapshot_calls[0].kwargs["node"]
    assert node.snapshot_source == "drive_state"
    assert node.metadata["dominant_drive"] == "coherence"
    assert node.metadata["tension_kinds"] == ["drive_competition.coherence_continuity"]


def test_materialization_without_audit_still_writes_state_snapshot(monkeypatch):
    """drive_audit may not have arrived yet (independent bus cadence) --
    materialization must still write pressures/activations, just without
    dominant_drive/summary/tension_kinds, rather than skip entirely."""
    worker = _make_worker(monkeypatch, materialization_enabled=True)
    store = MagicMock()
    worker._get_substrate_graph_store = MagicMock(return_value=store)
    worker._latest_drive_audit = None

    drive = _drive({"coherence": 0.72})
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=MagicMock(payload=drive.model_dump(mode="json")))
    worker._bus = bus
    worker._cache_drive_state_message({"data": b"payload"})

    snapshot_calls = [
        c for c in store.upsert_node.call_args_list if c.kwargs["node"].__class__.__name__ == "StateSnapshotNodeV1"
    ]
    assert len(snapshot_calls) == 1
    assert "dominant_drive" not in snapshot_calls[0].kwargs["node"].metadata


def test_materialization_fails_open_when_store_unavailable(monkeypatch):
    worker = _make_worker(monkeypatch, materialization_enabled=True)
    worker._get_substrate_graph_store = MagicMock(return_value=None)

    drive = _drive({"coherence": 0.72})
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=MagicMock(payload=drive.model_dump(mode="json")))
    worker._bus = bus
    # Must not raise.
    worker._cache_drive_state_message({"data": b"payload"})
    assert worker._latest_drive_state is not None


def test_materialization_fails_open_when_store_write_raises(monkeypatch):
    worker = _make_worker(monkeypatch, materialization_enabled=True)
    store = MagicMock()
    store.upsert_node.side_effect = RuntimeError("store down")
    worker._get_substrate_graph_store = MagicMock(return_value=store)

    drive = _drive({"coherence": 0.72})
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=MagicMock(payload=drive.model_dump(mode="json")))
    worker._bus = bus
    # Must not raise -- caching itself still succeeds even if every node write fails.
    worker._cache_drive_state_message({"data": b"payload"})
    assert worker._latest_drive_state is not None


def test_cache_drive_audit_message_fails_open_on_bad_decode(monkeypatch):
    worker = _make_worker(monkeypatch)
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=False, error="boom")
    worker._bus = bus

    worker._cache_drive_audit_message({"data": b"garbage"})
    assert worker._latest_drive_audit is None


def test_cache_drive_audit_message_caches_valid_audit(monkeypatch):
    worker = _make_worker(monkeypatch)
    audit = _audit()
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=MagicMock(payload=audit.model_dump(mode="json")))
    worker._bus = bus

    worker._cache_drive_audit_message({"data": b"payload"})
    assert worker._latest_drive_audit is not None
    assert worker._latest_drive_audit.dominant_drive == "coherence"


def test_materialize_no_op_when_no_drive_state_cached(monkeypatch):
    worker = _make_worker(monkeypatch, materialization_enabled=True)
    store = MagicMock()
    worker._get_substrate_graph_store = MagicMock(return_value=store)
    worker._latest_drive_state = None

    worker._materialize_drive_state_to_substrate()
    store.upsert_node.assert_not_called()
