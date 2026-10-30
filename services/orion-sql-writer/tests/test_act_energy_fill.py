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
    """Answers each of the module's statements from canned rows.

    ``rows``: (ts, chassis, cpu, cpu_idle, {gpuN_watts: w}) host readings. ``floors``: card key ->
    (value, n) idle floors."""

    def __init__(self, *, leases, rows, floors, card_leases=(), rate=None):
        self.leases, self.rows, self.floors = leases, rows, floors
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
        if stmt is fill._HOST_WINDOW_SQL:
            return _Result([(t, ch, c, idle, dict(cards)) for t, ch, c, idle, cards in self.rows])
        if stmt is fill._QUANTILE_SQL:
            return _Result([self.floors.get(params["key"], (None, 0))])
        raise AssertionError(f"unexpected statement {stmt}")


def readings(gpu2=140.0, gpu1=30.0, chassis=800.0, cpu=50.0, cpu_idle=50.0, span=(0, 181)):
    return [(_at(t), chassis, cpu, cpu_idle, {"gpu2_watts": gpu2, "gpu1_watts": gpu1})
            for t in range(span[0], span[1], 30)]


def rest_fixture(**hosts):
    from orion.physiology.rest_body import parse_rest_body

    hosts = hosts or {"circe": {"rest_w": 600, "approved_rest_w": 600}}
    return parse_rest_body({"version": 1, "hosts": hosts})


def heat_fixture(fitted_at):
    """A circe-only heat fit, independent of the committed config (which a refit rewrites)."""
    from orion.physiology.heat_config import parse_heat_sensitivity

    entry = {"point": 0.5, "ci95": [0.2, 0.8], "horizon": "2h", "n": 100, "fitted_at": fitted_at.isoformat(),
             "regressor": "compute_watts", "baseline_watts_p10": 200.0, "fit_command": "fixture"}
    return parse_heat_sensitivity({"schema": "heat_sensitivity.v1", "hosts": {"circe": entry}})


def usable_fixture_sensitivity():
    from orion.physiology.prices import usable_sensitivity

    return usable_sensitivity(0.5, 0.2, 0.8)


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    fill._FLOOR_CACHE.clear()
    fill._REST_CACHE.clear()
    monkeypatch.setattr(fill, "rest_body_config", lambda: (rest_fixture(), None))
    fill._POOL_CACHE.clear()
    fill._HEAT_CACHE.clear()
    monkeypatch.setattr(fill, "_card_locations", lambda: ({"gpu2": ("circe", 2), "gpu1": ("circe", 1)}, None))
    monkeypatch.setattr(fill, "_heat_config", lambda: (heat_fixture(T0 - timedelta(days=1)), None))


# gpu2 idle 40 W, gpu1 idle 30 W; chassis rest 600 W (pinned, rest_fixture); CPU idle 50 W (the row's).
FLOORS = {"gpu2_watts": (40.0, 2880), "gpu1_watts": (30.0, 2880)}
RATE = (0.1254564295292, datetime(2026, 10, 28, 7, tzinfo=timezone.utc), "rmp-ut-sch1-2026-08-10-r3")
LEASE = ("L1", '["gpu2"]', 60_000.0, _at(70))


def _cost(**kw):
    args = {"leases": [LEASE], "rows": readings(), "floors": FLOORS, "rate": RATE, **kw}
    conn = FakeConn(**args)
    return conn, fill.act_energy_cost(conn, correlation_id="c", since=_at(-3600), until=_at(200), now=_at(200))


def test_one_lease_alone_is_charged_its_share_of_the_whole_machine():
    """gpu2 100 W above idle is the host's only load; chassis 200 W above rest: the act pays 200 W."""
    conn, out = _cost()
    gpu_wh = 100.0 * 60.0 / 3600.0
    wh = 200.0 * 60.0 / 3600.0
    assert out["wh_by_host"] == {"circe": pytest.approx(wh, rel=1e-4)}
    assert out["wh_gpu_by_host"] == {"circe": pytest.approx(gpu_wh, rel=1e-4)}
    assert out["usd"] == pytest.approx(wh / 1000.0 * 0.1254564295292, rel=1e-4)
    # Heat is priced on the GPU part: the fit's regressor is compute watts.
    assert out["heat_c_by_host"] == {"circe": pytest.approx(usable_fixture_sensitivity() * gpu_wh / 100.0, rel=1e-3)}
    assert set(out["basis"]) == {"wh", "usd", "heat", "cpu"}
    assert out["basis"]["wh"].startswith(fill.WHOLE_MACHINE)
    assert "circe: whole machine above rest (chassis)" in out["basis"]["wh"] and "x2.00 of GPU" in out["basis"]["wh"]
    assert "share 1.00" in out["basis"]["wh"] and "idle 40.0 W" in out["basis"]["wh"]
    assert "chassis rest 600 W (pinned, rest_body.yaml)" in out["basis"]["wh"]
    # One host read, on its own host.
    assert [p["node"] for s, p in conn.calls if s is fill._HOST_WINDOW_SQL] == ["circe"]


def test_cpu_load_takes_its_share_of_the_overhead_and_is_not_charged():
    """CPU 100 W above its fitted idle + gpu2 100 W above idle; chassis 300 W above rest: the act
    pays 150 W."""
    _, out = _cost(rows=readings(chassis=900.0, cpu=150.0))
    assert out["wh_by_host"]["circe"] == pytest.approx(150.0 * 60.0 / 3600.0, rel=1e-4)
    assert out["wh_gpu_by_host"]["circe"] == pytest.approx(100.0 * 60.0 / 3600.0, rel=1e-4)


def test_a_shared_card_charges_this_lease_its_share():
    """Two other leases overlap this one on gpu2 the whole time: this act pays a third."""
    _, out = _cost(card_leases=[("L1", _at(10), _at(70), "c"), ("L2", _at(0), _at(120), "x"),
                                ("L3", _at(100), _at(150), "y"),  # after this lease: not overlapping
                                ("L4", _at(5), None, None)])      # still held at close: runs to now
    assert out["wh_gpu_by_host"]["circe"] == pytest.approx(100.0 * 60.0 / 3600.0 / 3.0, rel=1e-4)
    assert out["wh_by_host"]["circe"] == pytest.approx(200.0 * 60.0 / 3600.0 / 3.0, rel=1e-4)
    assert "share 0.33" in out["basis"]["wh"]


def test_no_lease_is_no_energy_and_no_basis():
    conn = FakeConn(leases=[], rows=readings(), floors=FLOORS)
    assert fill.act_energy_cost(conn, correlation_id="c", since=_at(0), until=_at(10)) == {}


@pytest.mark.parametrize("kwargs,reason", [
    ({"floors": {**FLOORS, "gpu2_watts": (40.0, 10)}}, "no_idle_floor:circe/gpu2_watts"),  # under 30 min of history
    ({"rows": [(_at(t), 800.0, 50.0, 50.0, {"gpu1_watts": 30.0}) for t in range(0, 181, 30)]},
     "no_watts_reading:circe/gpu2_watts"),                                              # card reported nothing
])
def test_unmeasurable_card_leaves_wh_absent_with_a_reason(kwargs, reason):
    _, out = _cost(**kwargs)
    assert "wh_by_host" not in out and "usd" not in out
    assert out["basis"]["wh"] == f"unmeasured: {reason}"


@pytest.mark.parametrize("kwargs,why", [
    ({"rows": readings(chassis=None)}, "no chassis_watts reading in the window"),
    ({"rest": rest_fixture(circe={"rest_w": 700, "approved_rest_w": 600})}, "rest_w 700 W > approved_rest_w 600 W"),
    ({"rest": rest_fixture(athena={"rest_w": 300, "approved_rest_w": 300})}, "no rest for circe in rest_body.yaml"),
    ({"floors": {**FLOORS, "gpu1_watts": (None, 0)}}, "no idle floor for card(s) [1]"),
    ({"rows": readings(cpu_idle=None)}, "no chassis, CPU (cpu_watts_total + cpu_idle_watts) or card reading"),
])
def test_no_whole_machine_split_falls_back_to_gpu_and_says_why(kwargs, why, monkeypatch):
    """hecate has no chassis meter today: its acts keep the GPU-above-idle number, named as such."""
    if "rest" in kwargs:
        rest = kwargs.pop("rest")
        monkeypatch.setattr(fill, "rest_body_config", lambda: (rest, None))
    _, out = _cost(**kwargs)
    gpu_wh = 100.0 * 60.0 / 3600.0
    assert out["wh_by_host"]["circe"] == pytest.approx(gpu_wh, rel=1e-4)
    assert out["wh_gpu_by_host"]["circe"] == pytest.approx(gpu_wh, rel=1e-4)
    assert f"circe: {fill.GPU_ONLY}, not whole machine (" in out["basis"]["wh"] and why in out["basis"]["wh"]


def test_a_pool_card_that_stops_reporting_is_a_hole_not_a_smaller_load():
    """Review finding: gpu1 is a pool card on circe with an idle floor but no reading in the window.
    Dropping it from the load would overcharge the act; the split falls back to GPU-only instead,
    the same way the hourly attribution marks those seconds split-unknown."""
    rows = [(_at(t), 800.0, 50.0, 50.0, {"gpu2_watts": 140.0}) for t in range(0, 181, 30)]
    _, out = _cost(rows=rows)
    assert out["wh_by_host"]["circe"] == pytest.approx(100.0 * 60.0 / 3600.0, rel=1e-4)
    assert f"circe: {fill.GPU_ONLY}" in out["basis"]["wh"]


def test_a_card_missing_from_the_pool_config_is_named():
    _, out = _cost(leases=[("L1", '["gpu9"]', 60_000.0, _at(70))])
    assert out == {"basis": {"cpu": fill.CPU_BASIS, "wh": "unmeasured: card_not_in_pool_config:gpu9"}}


def test_no_recent_rate_keeps_wh_and_says_why_usd_is_missing():
    _, out = _cost(rate=None)
    assert out["wh_by_host"]["circe"] > 0 and "usd" not in out
    assert out["basis"]["usd"].startswith("unmeasured")


def test_an_unfitted_host_has_no_heat_entry(monkeypatch):
    monkeypatch.setattr(fill, "_card_locations", lambda: ({"hecate-gpu0": ("hecate", 0)}, None))
    rows = [(_at(t), None, None, None, {"gpu0_watts": 140.0}) for t in range(0, 181, 30)]
    _, out = _cost(leases=[("L1", '["hecate-gpu0"]', 60_000.0, _at(70))], rows=rows,
                   floors={"gpu0_watts": (40.0, 2880)})
    assert "hecate" in out["wh_by_host"] and "heat_c_by_host" not in out
    assert "hecate unpriced (not_fitted:hecate)" in out["basis"]["heat"]
    assert f"hecate: {fill.GPU_ONLY}" in out["basis"]["wh"]


def test_the_committed_heat_fit_parses():
    """The real file only has to load; its numbers belong to fit_heat_sensitivity.py."""
    from orion.physiology.heat_config import load_heat_sensitivity

    cfg, err = load_heat_sensitivity(fill.HEAT_SENSITIVITY_PATH)
    assert err is None and "circe" in cfg.hosts


def test_interpolated_window_is_named_in_basis():
    """A 5 s lease between two 30 s samples has no sample inside it: basis says so."""
    _, out = _cost(leases=[("L1", '["gpu2"]', 5_000.0, _at(36))])
    assert "0 samples in 5 s (interpolated: no sample inside the lease)" in out["basis"]["wh"]


def test_a_grant_that_never_ends_stops_sharing_after_the_longest_hold():
    """Live: ~8 grants a day never get a terminal event. One from 3 h ago no longer shares the card."""
    _, out = _cost(card_leases=[("L9", _at(-9100), None, None)])
    assert "share 1.00" in out["basis"]["wh"]


def test_one_host_is_read_once_for_many_leases_and_floors_are_cached():
    conn, out = _cost(leases=[("L1", '["gpu2"]', 20_000.0, _at(40)), ("L2", '["gpu2"]', 20_000.0, _at(90)),
                              ("L3", '["gpu1"]', 20_000.0, _at(90))])
    assert out["wh_gpu_by_host"]["circe"] == pytest.approx(100.0 * 40.0 / 3600.0, rel=1e-4)
    assert sum(1 for s, _ in conn.calls if s is fill._HOST_WINDOW_SQL) == 1
    assert sum(1 for s, _ in conn.calls if s is fill._CARD_LEASES_SQL) == 2  # once per card
    keys = [p["key"] for s, p in conn.calls if s is fill._QUANTILE_SQL]
    assert sorted(keys) == sorted(set(keys))  # each floor read once
    conn2, again = _cost()
    assert again["wh_by_host"]  # second act, same day: floors from cache, not re-read
    assert not [p for s, p in conn2.calls if s is fill._QUANTILE_SQL]
    # card idle floors are day-keyed: the 24 h before the UTC day, not before the lease. Chassis
    # rest is never a quantile: it is pinned in rest_body.yaml.
    q = {p["key"]: p for s, p in conn.calls if s is fill._QUANTILE_SQL}
    assert q["gpu2_watts"]["until"] == "2026-10-29 00:00:00" and q["gpu2_watts"]["since"] == "2026-10-28 00:00:00"
    assert set(q) == {"gpu1_watts", "gpu2_watts"}


def test_the_real_pool_config_maps_cards_to_hosts():
    fill._POOL_CACHE.clear()
    cards, err = _REAL_CARD_LOCATIONS()
    assert err is None
    assert cards["gpu2"] == ("circe", 2) and cards["hecate-gpu0"] == ("hecate", 0)
