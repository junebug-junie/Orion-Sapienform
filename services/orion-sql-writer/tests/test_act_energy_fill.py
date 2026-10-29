"""Act energy fill on close (orion/grammar/act_energy_fill.py) against a scripted connection.

The SQL itself runs against real Postgres in test_act_ledger_integration.py (CI integration job);
this file pins the decisions: which card, which floor, which share, what is absent and why.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.grammar import act_energy_fill as fill

T0 = datetime(2026, 10, 29, 16, 40, 0, tzinfo=timezone.utc)
_REAL_CARD_LOCATIONS = fill._card_locations


def _at(sec):
    return T0 + timedelta(seconds=sec)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    """Answers each of the module's statements from canned rows."""

    def __init__(self, *, leases, watts, history, card_leases=(), rate=None):
        self.leases, self.watts, self.history = leases, watts, history
        self.card_leases, self.rate = list(card_leases), rate
        self.calls = []

    def execute(self, stmt, params):
        self.calls.append((stmt, params))
        if stmt is fill._ACT_LEASES_SQL:
            return _Result(self.leases)
        if stmt is fill._CARD_LEASES_SQL:
            return _Result(self.card_leases)
        if stmt is fill._RATE_SQL:
            return _Result([self.rate] if self.rate else [])
        if stmt is fill._CARD_WATTS_SQL:
            # The idle-floor read spans 24 h; the window read is the padded lease window.
            span = datetime.fromisoformat(params["until"].rstrip("~")) - datetime.fromisoformat(params["since"])
            return _Result(self.history if span > timedelta(hours=1) else self.watts)
        raise AssertionError(f"unexpected statement {stmt}")


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    fill._IDLE_CACHE.clear()
    fill._POOL_CACHE.clear()
    fill._HEAT_CACHE.clear()
    monkeypatch.setattr(fill, "_card_locations", lambda: ({"gpu2": ("circe", 2), "gpu1": ("circe", 1)}, None))


FLAT_140 = [(_at(t), 140.0) for t in range(0, 181, 30)]
IDLE_40 = [(_at(-3600 + i * 30), 40.0) for i in range(100)]
RATE = (0.1254564295292, datetime(2026, 10, 28, 7, tzinfo=timezone.utc), "rmp-ut-sch1-2026-08-10-r3")


def test_one_lease_alone_on_one_card():
    conn = FakeConn(leases=[("L1", '["gpu2"]', 60_000.0, _at(70))], watts=FLAT_140, history=IDLE_40, rate=RATE)
    out = fill.act_energy_cost(conn, correlation_id="c", since=_at(-3600), until=_at(200), now=_at(200))
    wh = 100.0 * 60.0 / 3600.0
    assert out["wh_by_host"] == {"circe": pytest.approx(wh, rel=1e-4)}
    assert out["usd"] == pytest.approx(wh / 1000.0 * 0.1254564295292, rel=1e-4)
    # The committed heat fit: circe 0.3111 C per 100 W.
    assert out["heat_c_by_host"] == {"circe": pytest.approx(0.3111 * wh / 100.0, rel=1e-3)}
    assert set(out["basis"]) == {"wh", "usd", "heat", "cpu"}
    assert "share 1.00" in out["basis"]["wh"] and "idle 40.0 W" in out["basis"]["wh"]
    # The card's own key, on its own host.
    keys = {(p["node"], p["key"]) for s, p in conn.calls if s is fill._CARD_WATTS_SQL}
    assert keys == {("circe", "gpu2_watts")}


def test_a_shared_card_charges_this_lease_its_share():
    """max_holds: 2 -- another lease held gpu2 the whole time: this act pays half."""
    conn = FakeConn(leases=[("L1", '["gpu2"]', 60_000.0, _at(70))], watts=FLAT_140, history=IDLE_40, rate=RATE,
                    card_leases=[("L1", _at(10), _at(70)), ("L2", _at(0), _at(120)),
                                 ("L3", _at(100), _at(150)),   # after this lease: not overlapping
                                 ("L4", _at(5), None)])        # still held at close: runs to now
    out = fill.act_energy_cost(conn, correlation_id="c", since=_at(-3600), until=_at(200), now=_at(200))
    assert out["wh_by_host"]["circe"] == pytest.approx(100.0 * 60.0 / 3600.0 / 3.0, rel=1e-4)
    assert "share 0.33" in out["basis"]["wh"]


def test_no_lease_is_no_energy_and_no_basis():
    conn = FakeConn(leases=[], watts=FLAT_140, history=IDLE_40)
    assert fill.act_energy_cost(conn, correlation_id="c", since=_at(0), until=_at(10)) == {}


@pytest.mark.parametrize("kwargs,reason", [
    ({"history": IDLE_40[:10]}, "no_idle_floor:circe/gpu2_watts"),       # under 30 min of card history
    ({"watts": []}, "no_watts_reading:circe/gpu2_watts"),               # card reported nothing
])
def test_unmeasurable_card_leaves_wh_absent_with_a_reason(kwargs, reason):
    args = {"leases": [("L1", '["gpu2"]', 60_000.0, _at(70))], "watts": FLAT_140, "history": IDLE_40, **kwargs}
    out = fill.act_energy_cost(FakeConn(**args), correlation_id="c", since=_at(0), until=_at(200), now=_at(200))
    assert "wh_by_host" not in out and "usd" not in out
    assert out["basis"]["wh"] == f"unmeasured: {reason}"


def test_a_card_missing_from_the_pool_config_is_named():
    conn = FakeConn(leases=[("L1", '["gpu9"]', 60_000.0, _at(70))], watts=FLAT_140, history=IDLE_40)
    out = fill.act_energy_cost(conn, correlation_id="c", since=_at(0), until=_at(200), now=_at(200))
    assert out == {"basis": {"cpu": fill.CPU_BASIS, "wh": "unmeasured: card_not_in_pool_config:gpu9"}}


def test_no_recent_rate_keeps_wh_and_says_why_usd_is_missing():
    conn = FakeConn(leases=[("L1", '["gpu2"]', 60_000.0, _at(70))], watts=FLAT_140, history=IDLE_40, rate=None)
    out = fill.act_energy_cost(conn, correlation_id="c", since=_at(0), until=_at(200), now=_at(200))
    assert out["wh_by_host"]["circe"] > 0 and "usd" not in out
    assert out["basis"]["usd"].startswith("unmeasured")


def test_an_unfitted_host_has_no_heat_entry(monkeypatch):
    monkeypatch.setattr(fill, "_card_locations", lambda: ({"hecate-gpu0": ("hecate", 0)}, None))
    conn = FakeConn(leases=[("L1", '["hecate-gpu0"]', 60_000.0, _at(70))], watts=FLAT_140, history=IDLE_40,
                    rate=RATE)
    out = fill.act_energy_cost(conn, correlation_id="c", since=_at(0), until=_at(200), now=_at(200))
    assert "hecate" in out["wh_by_host"] and "heat_c_by_host" not in out
    assert "hecate unpriced (not_fitted:hecate)" in out["basis"]["heat"]


def test_the_real_pool_config_maps_cards_to_hosts():
    fill._POOL_CACHE.clear()
    cards, err = _REAL_CARD_LOCATIONS()
    assert err is None
    assert cards["gpu2"] == ("circe", 2) and cards["hecate-gpu0"] == ("hecate", 0)
