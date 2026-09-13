"""`/health` surfaces whether cabinet/ambient RuntimeDirectory binds are readable.

Silent empty binds (Docker pinned to a deleted /run/orion-* inode after a
systemd RuntimeDirectory recreate) already caused a ~20h cabinet-key dropout
in orion_biometrics_summary (2026-09-12). Health must say so.
"""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault(
    "NODE_CATALOG_PATH",
    str(pathlib.Path(__file__).resolve().parents[3] / "config" / "biometrics" / "node_catalog.yaml"),
)

from app.main import _host_snapshot_health  # noqa: E402
import app.main as m  # noqa: E402


def test_host_snapshot_health_readable_when_files_exist(tmp_path, monkeypatch):
    cabinet = tmp_path / "latest.json"
    ambient = tmp_path / "audio.json"
    cabinet.write_text("{}", encoding="utf-8")
    ambient.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m.settings, "CABINET_SENSORS_PATH", str(cabinet))
    monkeypatch.setattr(m.settings, "AMBIENT_AUDIO_PATH", str(ambient))

    body = _host_snapshot_health()
    assert body["cabinet_sensors"]["readable"] is True
    assert body["ambient_audio"]["readable"] is True
    assert body["cabinet_sensors"]["reason"] is None


def test_host_snapshot_health_flags_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(m.settings, "CABINET_SENSORS_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(m.settings, "AMBIENT_AUDIO_PATH", str(tmp_path / "also-missing.json"))

    body = _host_snapshot_health()
    assert body["cabinet_sensors"]["readable"] is False
    assert body["ambient_audio"]["readable"] is False
    assert body["cabinet_sensors"]["reason"] == "missing_or_unreadable"


def test_health_endpoint_includes_host_snapshots_without_flipping_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(m.settings, "CABINET_SENSORS_PATH", str(tmp_path / "gone.json"))
    ambient = tmp_path / "audio.json"
    ambient.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m.settings, "AMBIENT_AUDIO_PATH", str(ambient))

    payload = m.health()
    # Optional instruments must not degrade top-level health (non-athena nodes
    # never have cabinet/ambient files).
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["host_snapshots"]["cabinet_sensors"]["readable"] is False
    assert payload["host_snapshots"]["ambient_audio"]["readable"] is True
