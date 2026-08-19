"""ROADMAP A5 -- the admission ledger, and the one distinction it exists to make.

The load-bearing test here is `test_first_poll_admit_is_not_a_deferral`. Live on 2026-08-19,
294 of 294 background admissions cleared on the first poll with `waited` between 0.012s and
0.091s -- the HTTP round trip of asking `/slots`, not a wait. A ledger that counted those as
deferrals would report ~300 phantom waits a day into Orion's context.
"""
from __future__ import annotations

import dataclasses

import pytest

from app.admission_ledger import AdmissionLedger, AdmissionRecord, get_ledger


def _rec(ledger: AdmissionLedger, *, ts: float, polls: int = 1, waited: float = 0.02,
         outcome: str = "admitted", route: str = "quick_background") -> AdmissionRecord:
    return ledger.record(
        route_key=route, url="http://atlas:8013", waited_s=waited,
        polls=polls, reserved=2, outcome=outcome, ts=ts,
    )


def test_first_poll_admit_is_not_a_deferral():
    """The whole point. One poll means the answer was yes on the first ask."""
    led = AdmissionLedger()
    for i in range(294):
        _rec(led, ts=1000.0 + i, polls=1, waited=0.021)
    snap = led.snapshot(window_s=3600.0, now=1300.0)
    assert snap["deferrals"] == 0
    assert snap["longest_wait_s"] == 0.0
    assert snap["deferred_s_total"] == 0.0
    # ...but the window is not empty, and that is the fact that makes the zero readable.
    assert snap["checked"] > 0


def test_a_slept_poll_interval_is_a_deferral():
    led = AdmissionLedger()
    _rec(led, ts=1000.0, polls=1, waited=0.02)
    _rec(led, ts=1001.0, polls=5, waited=2.4)
    snap = led.snapshot(window_s=3600.0, now=1002.0)
    assert snap["deferrals"] == 1
    assert snap["longest_wait_s"] == 2.4
    assert snap["deferred_s_total"] == 2.4
    assert snap["last_deferral_ts"] == 1001.0


def test_timeout_forwarded_is_a_deferral_even_at_one_poll():
    """A timeout at polls=1 is pathological but must not be silently dropped."""
    led = AdmissionLedger()
    _rec(led, ts=1000.0, polls=1, waited=30.0, outcome="timeout_forwarded")
    snap = led.snapshot(window_s=3600.0, now=1001.0)
    assert snap["deferrals"] == 1
    assert snap["timeouts"] == 1
    assert snap["longest_wait_s"] == 30.0


def test_unchecked_is_counted_but_is_not_a_deferral():
    """/slots unreachable means the gate failed open -- nothing waited, and nothing is known."""
    led = AdmissionLedger()
    _rec(led, ts=1000.0, polls=1, waited=0.02, outcome="unchecked")
    snap = led.snapshot(window_s=3600.0, now=1001.0)
    assert snap["checked"] == 1
    assert snap["unchecked"] == 1
    assert snap["deferrals"] == 0


def test_empty_window_is_distinguishable_from_a_quiet_one():
    """`deferrals == 0` alone is ambiguous; `checked` is what disambiguates it."""
    led = AdmissionLedger()
    empty = led.snapshot(window_s=3600.0, now=1000.0)
    assert empty == {
        "window_s": 3600.0, "checked": 0, "deferrals": 0, "timeouts": 0, "unchecked": 0,
        "deferred_s_total": 0.0, "longest_wait_s": 0.0, "last_deferral_ts": None, "routes": [],
    }
    # An empty `sum()` returns int 0, so without the float() cast this field would change TYPE
    # between an idle and a busy window -- an int at rest, a float once anything was deferred.
    assert isinstance(empty["deferred_s_total"], float)
    for i in range(10):
        _rec(led, ts=1000.0 + i, polls=1)
    quiet = led.snapshot(window_s=3600.0, now=1010.0)
    assert quiet["deferrals"] == 0 and quiet["checked"] == 10
    assert empty["checked"] != quiet["checked"]


def test_window_excludes_older_records():
    led = AdmissionLedger()
    _rec(led, ts=1000.0, polls=9, waited=8.0)      # old deferral
    _rec(led, ts=9000.0, polls=1, waited=0.02)     # recent admit
    snap = led.snapshot(window_s=600.0, now=9001.0)
    assert snap["checked"] == 1
    assert snap["deferrals"] == 0


def test_ledger_is_bounded():
    led = AdmissionLedger(max_records=16)
    for i in range(200):
        _rec(led, ts=1000.0 + i)
    assert led.snapshot(window_s=86400.0, now=1200.0)["checked"] == 16


def test_routes_are_reported():
    led = AdmissionLedger()
    _rec(led, ts=1000.0, route="quick_background")
    _rec(led, ts=1001.0, route="metacog_background")
    assert led.snapshot(window_s=600.0, now=1002.0)["routes"] == [
        "metacog_background", "quick_background",
    ]


def test_ledger_holds_no_request_content():
    """Pins the privacy boundary from the module docstring: timings only, structurally.

    Not a policy comment -- the record type's own fields are the guarantee, so a future field
    carrying a prompt, a response, or a user id fails here rather than shipping.
    """
    fields = {f.name for f in dataclasses.fields(AdmissionRecord)}
    assert fields == {"ts", "route_key", "url", "waited_s", "polls", "reserved", "outcome"}


def test_negative_and_junk_inputs_are_normalised():
    led = AdmissionLedger()
    r = led.record(route_key=None, url=None, waited_s=-5.0, polls=-3, reserved=-1,
                   outcome=None, ts=1000.0)
    assert (r.waited_s, r.polls, r.reserved, r.route_key, r.url, r.outcome) == (
        0.0, 0, 0, "", "", "")
    assert r.is_deferral is False


def test_get_ledger_is_the_same_process_wide_instance():
    assert get_ledger() is get_ledger()


@pytest.mark.parametrize("window,expected", [(0.0, 1), (-10.0, 1), (1.0, 2), (86400.0, 2)])
def test_a_negative_window_is_clamped_to_zero_not_run_backwards(window, expected):
    """Hand-computed, not read off the implementation.

    cutoff = now - max(0, window). With now=1000.0 the boundary is inclusive (`ts >= cutoff`):

        window  0.0  -> cutoff 1000.0 -> {1000.0}          -> 1
        window -10.0 -> clamped to 0  -> same as above     -> 1   (NOT 990.0, which would be 2)
        window  1.0  -> cutoff  999.0 -> {999.0, 1000.0}   -> 2

    The -10.0 case is the one that matters: without the clamp a negative window would widen the
    cutoff into the past and quietly report MORE history than asked for.
    """
    led = AdmissionLedger()
    _rec(led, ts=999.0)
    _rec(led, ts=1000.0)
    assert led.snapshot(window_s=window, now=1000.0)["checked"] == expected
