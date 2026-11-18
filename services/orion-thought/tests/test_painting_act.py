"""A durable painting run is one `make` act (act ledger, rung 2).

the first real generation opens it exactly once (2026-10-21: no longer prepare -- a run that
never reaches the GPU is not an attempt; covered against the step store in test_visual_steps.py);
produced closes 1; a terminal or a non-produced abandon closes 0 only if the act was opened;
retries publish nothing. The published events are
valid act grammar events.
"""
from __future__ import annotations

import pytest

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reverie_visual import VisualRunRequestV1
from orion.schemas.reverie_visual_run import ReverieVisualStepRequestV1, ReverieVisualStepResultV1

from app import visual_steps

REQUEST = VisualRunRequestV1(dispatch_id="dispatch-1", proposal_id="proposal-1")


def _req(step: str) -> ReverieVisualStepRequestV1:
    return ReverieVisualStepRequestV1(run_id="run-1", correlation_id="corr-1", step=step,
                                      visual_request=REQUEST, attempt_id=None if step in ("prepare", "abandon") else "a1")


def _res(step: str, status: str, **kw) -> ReverieVisualStepResultV1:
    kw.setdefault("attempt_id", "a1")
    return ReverieVisualStepResultV1(run_id="run-1", correlation_id="corr-1", step=step, status=status,
                                     elapsed_sec=1.0, **kw)


@pytest.mark.parametrize("step,result,expected", [
    # Never "open" from a result: prepare_step opens once, on its first plan write.
    ("prepare", _res("prepare", "done"), ("none", None)),
    ("prepare", _res("prepare", "retry", reason="x"), ("none", None)),
    ("prepare", _res("prepare", "terminal", outcome="failed", reason="ineligible"), ("none", None)),
    ("prepare", _res("prepare", "terminal", outcome="already_satisfied", reason="x"), ("none", None)),
    # A replay on an attempt that already produced: terminal/produced must grade 1, not 0.
    ("caption", _res("caption", "terminal", outcome="produced", reason="attempt_produced"), ("close", True)),
    ("prepare", _res("prepare", "terminal", outcome="produced", reason="attempt_produced"), ("close", True)),
    ("caption", _res("caption", "retry", reason="needs_generate"), ("none", None)),
    ("caption", _res("caption", "terminal", outcome="failed", reason="retry_window_expired"), ("close", False)),
    ("caption", _res("caption", "done", outcome="produced", reason="ok"), ("close", True)),
    ("abandon", _res("abandon", "done", outcome="unknown", reason="run_abandoned"), ("close", False)),
    ("abandon", _res("abandon", "done", outcome="produced", reason="attempt_already_produced"), ("close", True)),
])
def test_transition_table(step, result, expected):
    action, produced, _note = visual_steps.painting_act_transition(_req(step), result)
    assert (action, produced) == expected


class _Bus:
    def __init__(self):
        self.sent = []

    async def publish(self, channel, envelope):
        self.sent.append((channel, envelope))


async def _drive(monkeypatch, step, result, enabled=True):
    async def inner(bus, req, **kw):
        return result

    monkeypatch.setattr(visual_steps, "_run_visual_step_inner", inner)
    monkeypatch.setattr(visual_steps.settings, "thought_act_ledger_enabled", enabled, raising=False)
    bus = _Bus()
    out = await visual_steps.run_visual_step(bus, _req(step))
    assert out is result
    return [GrammarEventV1.model_validate(env.payload) for ch, env in bus.sent if ch == "orion:grammar:event"]


@pytest.mark.asyncio
async def test_open_has_an_honest_expectation_and_close_shares_its_act_id(monkeypatch):
    bus = _Bus()
    monkeypatch.setattr(visual_steps.settings, "thought_act_ledger_enabled", True, raising=False)
    await visual_steps._publish_painting_open(bus, _req("prepare"))
    opened = [GrammarEventV1.model_validate(env.payload) for ch, env in bus.sent if ch == "orion:grammar:event"]
    assert [e.event_kind for e in opened] == ["act_opened"]
    act = opened[0].act
    assert act.verb == "make" and act.actor == "thought.painting" and act.level == "deliberate"
    assert act.expect.signal_id == "artifact_produced"
    assert act.expect.horizon_sec == visual_steps.PAINTING_HORIZON_SEC and "deadline" in act.expect.basis
    assert opened[0].provenance.source_service == "orion-thought"
    closed = await _drive(monkeypatch, "caption", _res("caption", "done", outcome="produced", reason="ok"))
    assert [e.event_kind for e in closed] == ["act_closed"]
    assert closed[0].act.act_id == act.act_id
    assert closed[0].act.outcome.value == 1.0


@pytest.mark.asyncio
async def test_retry_publishes_nothing_and_flag_off_publishes_nothing(monkeypatch):
    assert await _drive(monkeypatch, "caption", _res("caption", "retry", reason="needs_generate")) == []
    assert await _drive(monkeypatch, "prepare", _res("prepare", "done")) == []
    assert await _drive(monkeypatch, "caption", _res("caption", "done", outcome="produced", reason="ok"),
                        enabled=False) == []
    monkeypatch.setattr(visual_steps.settings, "thought_act_ledger_enabled", False, raising=False)
    bus = _Bus()
    await visual_steps._publish_painting_open(bus, _req("prepare"))
    assert bus.sent == []


@pytest.mark.asyncio
async def test_bus_failure_never_fails_the_step(monkeypatch):
    class Broken:
        async def publish(self, *a, **k):
            raise RuntimeError("bus down")

    result = _res("prepare", "done")

    async def inner(bus, req, **kw):
        return result

    monkeypatch.setattr(visual_steps, "_run_visual_step_inner", inner)
    assert await visual_steps.run_visual_step(Broken(), _req("prepare")) is result


def test_flag_ships_on():
    from app.settings import ThoughtSettings

    assert ThoughtSettings.model_fields["thought_act_ledger_enabled"].default is True
