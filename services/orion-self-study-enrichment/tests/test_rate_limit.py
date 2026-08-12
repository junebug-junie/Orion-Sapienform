from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.rate_limit import allow_and_record, read_state  # noqa: E402


def test_allow_and_record_respects_ceiling(tmp_path):
    path = tmp_path / "state.json"
    assert allow_and_record(path, max_per_day=2) is True
    assert allow_and_record(path, max_per_day=2) is True
    assert allow_and_record(path, max_per_day=2) is False


def test_allow_and_record_disabled_when_zero(tmp_path):
    path = tmp_path / "state.json"
    assert allow_and_record(path, max_per_day=0) is False


def test_read_state_missing_file_defaults_to_today_zero(tmp_path):
    state = read_state(tmp_path / "nope.json")
    assert state.count == 0
