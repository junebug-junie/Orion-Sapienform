"""`/health` says whether this node's power telemetry is actually wired.

Both power pollers are OPTIONAL and both go quiet by design -- a node with no BMC, or no
metered outlets, reports nothing rather than a fake zero. That is right for the DATA and it is
exactly what makes the INSTRUMENT undiagnosable: "configured and working", "configured but
unreachable", and "running an image that predates the feature" all look identical from outside
the container.

Paid for twice. circe's BMC has been unreachable long enough that nobody knows when it broke.
And the day the PDU poller shipped, answering "is atlas configured, failing, or on an old
image?" needed a `docker exec` on a remote host, because nothing exposed it.

The distinction these tests pin: `healthy` is TRISTATE. False means broken. None means either
"not applicable" or "has not reported yet" -- and conflating those with False is how a health
check cries wolf on every restart and then gets ignored.
"""
from __future__ import annotations

import os
import pathlib

# app.main loads the node catalog at import time from a CONTAINER path. Point it at the repo's
# real config before that import happens, or collection fails on the filesystem rather than on
# anything this file is testing.
os.environ.setdefault(
    "NODE_CATALOG_PATH",
    str(pathlib.Path(__file__).resolve().parents[3] / "config" / "biometrics" / "node_catalog.yaml"),
)

import pytest

from app.pdu import PduPoller


class _FakeIlo:
    def __init__(self, enabled, detail):
        self.enabled = enabled
        self._detail = detail

    def details(self):
        return self._detail


def _health(monkeypatch, *, ilo, pdu, outlets="", host=""):
    import app.main as m

    monkeypatch.setattr(m, "_ilo_poller", ilo)
    monkeypatch.setattr(m, "_pdu_poller", pdu)
    monkeypatch.setattr(m.settings, "PDU_OUTLETS", outlets)
    monkeypatch.setattr(m.settings, "PDU_HOST", host)
    return m._power_telemetry_health()


# ------------------------------------------------------------ not configured is not a fault

def test_a_node_with_neither_reports_not_configured_not_broken(monkeypatch):
    """athena has no PDU outlets; most nodes have no BMC. Neither is a failure, and neither
    should read as one."""
    h = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=PduPoller("", []))
    assert h["ilo"] == {"configured": False, "healthy": None, "reason": "not_configured"}
    assert h["pdu"]["configured"] is False
    assert h["pdu"]["healthy"] is None
    assert h["pdu"]["reason"] == "not_configured"


def test_healthy_is_never_false_merely_for_being_absent(monkeypatch):
    """The whole point of the tristate. Absent must not read as broken."""
    h = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=PduPoller("", []))
    assert h["ilo"]["healthy"] is not False
    assert h["pdu"]["healthy"] is not False


# ------------------------------------------------------------ configured and broken

def test_an_unreachable_pdu_is_reported_as_unhealthy_with_the_reason(monkeypatch):
    """The case that cost a round trip: configured, deployed, and silently failing."""
    p = PduPoller("192.168.1.39", [34, 35])
    p._snapshot = type(p._snapshot)(error="timeout")
    h = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=p,
                outlets="34,35", host="192.168.1.39")
    assert h["pdu"]["configured"] is True
    assert h["pdu"]["healthy"] is False
    assert h["pdu"]["reason"] == "timeout"


def test_an_unreachable_bmc_is_reported_too(monkeypatch):
    """circe's BMC. Unreachable for an unknown length of time precisely because nothing said so."""
    h = _health(monkeypatch,
                ilo=_FakeIlo(True, {"ilo_error": "No route to host"}),
                pdu=PduPoller("", []))
    assert h["ilo"]["configured"] is True
    assert h["ilo"]["healthy"] is False
    assert "No route to host" in h["ilo"]["reason"]


# ------------------------------------------------------------ configured and working

def test_a_working_pdu_reports_healthy(monkeypatch):
    p = PduPoller("192.168.1.39", [19, 25, 31])
    p._snapshot = type(p._snapshot)(watts=419.0, per_outlet_watts={19: 129.0})
    h = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=p,
                outlets="19,25,31", host="192.168.1.39")
    assert h["pdu"]["healthy"] is True
    assert h["pdu"]["reason"] is None


def test_a_working_bmc_reports_healthy(monkeypatch):
    h = _health(monkeypatch,
                ilo=_FakeIlo(True, {"ilo_power_watts": 470.0}),
                pdu=PduPoller("", []))
    assert h["ilo"]["healthy"] is True


# ------------------------------------------------------------ started but not yet reported

def test_no_reading_yet_is_none_not_false(monkeypatch):
    """Every restart passes through this state. Reporting it as broken would make the check
    fire on every deploy and therefore get ignored -- which is worse than not having it."""
    p = PduPoller("192.168.1.39", [34, 35])
    h = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=p,
                outlets="34,35", host="192.168.1.39")
    assert h["pdu"]["configured"] is True
    assert h["pdu"]["healthy"] is None
    assert h["pdu"]["reason"] == "no_reading_yet"


# ------------------------------------------------------------ the outlet map is echoed

def test_the_outlet_map_is_visible_without_shelling_into_the_host(monkeypatch):
    """The map is physical cabling that nothing can verify automatically, so at minimum it
    should be readable remotely -- a wrong map is otherwise invisible until the watts look odd."""
    h = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=PduPoller("192.168.1.39", [19, 25, 31]),
                outlets="19,25,31", host="192.168.1.39")
    assert h["pdu"]["outlets"] == [19, 25, 31]
    assert h["pdu"]["host"] == "192.168.1.39"


def test_an_unset_host_reads_as_none_not_empty_string(monkeypatch):
    h = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=PduPoller("", []))
    assert h["pdu"]["host"] is None
    assert h["pdu"]["outlets"] == []


# ------------------------------------------------------------ the three states are distinct

def test_the_three_confusable_states_are_now_distinguishable(monkeypatch):
    """The actual deliverable: old-image, configured-but-broken and working are three different
    answers, where before all three were an identical silence."""
    old_image = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=PduPoller("", []))["pdu"]

    broken_p = PduPoller("192.168.1.39", [34, 35])
    broken_p._snapshot = type(broken_p._snapshot)(error="No route to host")
    broken = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=broken_p,
                     outlets="34,35", host="192.168.1.39")["pdu"]

    ok_p = PduPoller("192.168.1.39", [34, 35])
    ok_p._snapshot = type(ok_p._snapshot)(watts=266.0)
    working = _health(monkeypatch, ilo=_FakeIlo(False, {}), pdu=ok_p,
                      outlets="34,35", host="192.168.1.39")["pdu"]

    signatures = {
        (old_image["configured"], old_image["healthy"]),
        (broken["configured"], broken["healthy"]),
        (working["configured"], working["healthy"]),
    }
    assert len(signatures) == 3
