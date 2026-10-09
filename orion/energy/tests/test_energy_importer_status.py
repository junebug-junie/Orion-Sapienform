from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.energy.importer_status import (
    PortalStatus,
    compute_importer_status,
    parse_portal_status,
    portal_status_dict,
)

NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


def _status(**over):
    base = dict(
        portal_enabled=False, portal=None, portal_interval_hours=24.0,
        latest_interval_end=NOW - timedelta(hours=30), last_file_at=NOW - timedelta(hours=2),
        now=NOW, stale_after_hours=48.0,
    )
    base.update(over)
    return compute_importer_status(**base)


def test_file_drop_fresh_usage_is_healthy() -> None:
    s = _status()
    assert (s.state, s.reason, s.source) == ("healthy", "usage_fresh", "file_drop")
    assert s.usage_lag_hours == pytest.approx(30.0)
    assert s.last_success_at == NOW - timedelta(hours=2)


def test_old_usage_is_stale() -> None:
    s = _status(latest_interval_end=NOW - timedelta(hours=50))
    assert s.state == "stale" and s.reason.startswith("usage_lag_hours=50.0")


def test_no_usage_is_stale_not_zero() -> None:
    s = _status(latest_interval_end=None)
    assert (s.state, s.reason, s.usage_lag_hours) == ("stale", "no_usage_yet", None)


def _portal(state="ok", attempt_hours_ago=1.0, reason=None):
    return PortalStatus(
        state=state, reason=reason or state,
        last_attempt_at=NOW - timedelta(hours=attempt_hours_ago), last_success_at=NOW - timedelta(hours=26),
    )


def test_reauth_beats_fresh_usage() -> None:
    s = _status(portal_enabled=True, portal=_portal("reauth_required", reason="session_expired"))
    assert (s.state, s.reason, s.source) == ("reauth_required", "session_expired", "portal")


def test_portal_missing_status_is_degraded() -> None:
    s = _status(portal_enabled=True, portal=None)
    assert s.state == "degraded"
    assert s.reason == "portal_status_missing"


def test_portal_error_is_degraded_with_its_reason() -> None:
    s = _status(portal_enabled=True, portal=_portal("error", reason="empty_download"))
    assert (s.state, s.reason) == ("degraded", "empty_download")


def test_portal_not_running_is_degraded() -> None:
    s = _status(portal_enabled=True, portal=_portal("ok", attempt_hours_ago=72.0))
    assert (s.state, s.reason) == ("degraded", "portal_not_running")


def test_portal_ok_uses_portal_times() -> None:
    s = _status(portal_enabled=True, portal=_portal("ok"))
    assert s.state == "healthy"
    assert s.last_success_at == NOW - timedelta(hours=26)
    assert s.last_attempt_at == NOW - timedelta(hours=1)


def test_portal_not_running_boundary_still_healthy() -> None:
    s = _status(portal_enabled=True, portal=_portal("ok", attempt_hours_ago=30.0))
    assert s.state == "healthy"


def test_portal_ok_stale_usage_is_stale() -> None:
    s = _status(
        portal_enabled=True, portal=_portal("ok"),
        latest_interval_end=NOW - timedelta(hours=50),
    )
    assert s.state == "stale"
    assert s.reason.startswith("usage_lag_hours=50.0")


def test_parse_portal_status_roundtrip_and_rejects_unknown() -> None:
    p = _portal("error", reason="timeout")
    assert parse_portal_status(portal_status_dict(p)) == p
    with pytest.raises(ValueError):
        parse_portal_status({"state": "fine", "last_attempt_at": NOW.isoformat()})
    with pytest.raises(ValueError):
        parse_portal_status({"state": "ok"})
