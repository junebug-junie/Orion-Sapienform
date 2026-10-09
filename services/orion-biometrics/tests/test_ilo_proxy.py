"""BMC proxy polling: athena reads hecate's BMC on hecate's behalf (ILO_PROXY_NODES)."""
from __future__ import annotations

import json

from app import main
from app.ilo import IloPoller, IloSnapshot, parse_proxy_bmcs

_OK = {"host": "https://192.168.1.75", "username": "ro", "password": "pw"}


def test_parse_valid_entry_lowercases_node():
    assert parse_proxy_bmcs(json.dumps({"Hecate": _OK})) == {"hecate": _OK}


def test_parse_empty_and_garbage_disable_proxying():
    assert parse_proxy_bmcs("") == {}
    assert parse_proxy_bmcs("{not json") == {}
    assert parse_proxy_bmcs("[1,2]") == {}


def test_parse_drops_incomplete_entries():
    raw = json.dumps({"hecate": {"host": "https://x", "username": "u"}, "circe": _OK})
    assert list(parse_proxy_bmcs(raw)) == ["circe"]


def _poller_with_watts(watts):
    p = IloPoller(_OK["host"], _OK["username"], _OK["password"])
    p._snapshot = IloSnapshot(fetched_at=1.0, power_watts=watts)
    return p


def test_bmc_watts_fill_chassis_watts_for_proxied_node(monkeypatch):
    monkeypatch.setattr(main, "_ilo_proxy_pollers", {"hecate": _poller_with_watts(612.0)})
    monkeypatch.setattr(main, "_pdu_proxy_pollers", {})
    assert main._proxy_measurements() == {"hecate": {"chassis_watts": 612.0}}


def test_pdu_reading_wins_over_bmc(monkeypatch):
    class _Pdu:
        def details(self):
            return {"pdu_watts": 700.0}

    monkeypatch.setattr(main, "_ilo_proxy_pollers", {"hecate": _poller_with_watts(612.0)})
    monkeypatch.setattr(main, "_pdu_proxy_pollers", {"hecate": _Pdu()})
    out = main._proxy_measurements()
    assert out["hecate"] == {"chassis_watts": 700.0, "pdu_watts": 700.0}


def test_failed_or_unpolled_bmc_contributes_nothing(monkeypatch):
    bad = IloPoller(_OK["host"], _OK["username"], _OK["password"])
    bad._snapshot = IloSnapshot(fetched_at=1.0, error="timeout")
    monkeypatch.setattr(main, "_ilo_proxy_pollers", {"hecate": bad})
    monkeypatch.setattr(main, "_pdu_proxy_pollers", {})
    assert main._proxy_measurements() == {}
