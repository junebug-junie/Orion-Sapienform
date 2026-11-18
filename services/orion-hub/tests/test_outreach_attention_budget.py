"""One attention budget (2026-11-10): sends Juniper asked for are exempt from the daily cap."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from test_outreach_door_policy import _endogenous, _loop, _Outreach, _reach


def test_an_asked_for_send_passes_a_full_cap_and_does_not_count(monkeypatch):
    import scripts.endogenous_outreach as eo

    o, rows, delivered = _endogenous(monkeypatch)
    o.daily_cap = 2
    o._gate_inputs()
    o._sent_today, o._sent_today_recovered = 2, True
    assert eo.outreach_block_reason(o._gate_inputs(), skip_schedule_gates=True) == "daily_cap"
    out = asyncio.run(o.offer_message(text="The backup ran.\nWhy Juniper: you asked.", correlation_id="c",
                                      tag="curiosity_outreach", skip_schedule_gates=True,
                                      require_why_juniper=True, cap_exempt=True,
                                      meta={"door_policy": "asked_by_juniper"}))
    assert out["reason"] == "sent" and delivered == ["The backup ran."] and o._sent_today == 2
    assert rows[-1]["door_policy"] == "asked_by_juniper"
    # The same message without the exemption is held by the cap.
    held = asyncio.run(o.offer_message(text="Unasked.\nWhy Juniper: x.", correlation_id="c2",
                                       tag="curiosity_outreach", skip_schedule_gates=True,
                                       require_why_juniper=True))
    assert held["reason"] == "daily_cap"


class _CapOutreach(_Outreach):
    def __init__(self):
        super().__init__()
        self.pre_checks: list[dict] = []

    def blocked_reason(self, **kw):
        self.pre_checks.append(kw)
        return None if kw.get("cap_exempt") else "daily_cap"


def test_a_juniper_asked_curiosity_run_is_offered_cap_exempt(monkeypatch):
    outreach = _CapOutreach()
    loop = _loop(monkeypatch, outreach, focus=SimpleNamespace(origin="juniper", chat_correlation_id="chat-1"))
    assert _reach(loop, "investigate") is None
    assert outreach.pre_checks[-1].get("cap_exempt") is True
    assert outreach.offered[0]["cap_exempt"] is True


def test_an_unasked_self_inquiry_run_is_held_by_the_cap(monkeypatch):
    outreach = _CapOutreach()
    loop = _loop(monkeypatch, outreach)
    assert _reach(loop, "self_inquiry") == "daily_cap" and outreach.offered == []


def test_count_sent_on_and_the_body_skip_asked_for_sends():
    import inspect

    from scripts import endogenous_outreach_decisions as d

    src = inspect.getsource(d.count_sent_on)
    assert "door_policy" in src and "CAP_EXEMPT_DOOR_POLICIES" in src
