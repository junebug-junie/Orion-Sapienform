from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_journal_dispatch_registry as gate  # noqa: E402


def test_find_unregistered_trigger_kinds_empty_when_fully_covered() -> None:
    trigger_to_mode = {"daily_summary": "daily", "manual": "manual"}
    registry = {"daily_summary": object(), "manual": object()}
    assert gate.find_unregistered_trigger_kinds(trigger_to_mode, registry) == []


def test_find_unregistered_trigger_kinds_reports_gap() -> None:
    trigger_to_mode = {"daily_summary": "daily", "brand_new_kind": "digest"}
    registry = {"daily_summary": object()}
    assert gate.find_unregistered_trigger_kinds(trigger_to_mode, registry) == ["brand_new_kind"]


def test_find_orphaned_registry_entries_reports_retired_row() -> None:
    trigger_to_mode = {"daily_summary": "daily"}
    registry = {"daily_summary": object(), "retired_kind": object()}
    assert gate.find_orphaned_registry_entries(trigger_to_mode, registry) == ["retired_kind"]


def test_main_passes_against_the_real_registry() -> None:
    """Regression/live check: the actual orion.journaler.worker._TRIGGER_TO_MODE and
    orion.journaler.dispatch_registry.JOURNAL_DISPATCH_REGISTRY, as shipped."""
    exit_code = gate.main([])
    assert exit_code == 0


def test_main_json_output_shape_against_the_real_registry(capsys) -> None:
    exit_code = gate.main(["--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["unregistered_trigger_kinds"] == []
    assert payload["trigger_kind_count"] > 0
    assert payload["registry_entry_count"] > 0


def test_main_fails_when_a_trigger_kind_is_unregistered(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "_load",
        lambda: (
            {"daily_summary": "daily", "brand_new_kind": "digest"},
            {"daily_summary": object()},
        ),
    )
    exit_code = gate.main([])
    assert exit_code == 1


def test_main_reports_missing_trigger_kind_in_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        gate,
        "_load",
        lambda: (
            {"daily_summary": "daily", "brand_new_kind": "digest"},
            {"daily_summary": object()},
        ),
    )
    exit_code = gate.main([])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "brand_new_kind" in out


def test_main_import_failure_exits_two(monkeypatch) -> None:
    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(gate, "_load", _raise)
    exit_code = gate.main([])
    assert exit_code == 2
