"""Intention ledger D1/D2 on the Hub side (shadow): the judge's view, the promise acts, and the motor's blindness.

* the view: open promise acts -> (act id, quote read from the promising turn's record); the previous
  reply's offers -> (ref, quote), firm items only;
* the acts: a delivered Juniper turn opens one per firm promise, one per accepted earlier offer, and
  closes each kept promise; unsolicited rows, other sources and the flags off publish nothing;
* acceptance check 5: open promises ride HarnessRunRequestV1 to the judge, never into the motor prompt.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.grammar.promise_acts import promise_act_id, promise_ref
from orion.grammar.promise_view import offers_from_made, view_from_rows
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.harness_finalize import (
    FinalizeReflectionV1,
    HarnessRunRequestV1,
    OpenPromiseRefV1,
    PromiseKeptV1,
    PromiseV1,
)

REPO = Path(__file__).resolve().parents[3]
QUOTE = "I'll hold the deep read until the pipeline digests it, then pull it back."
DONE = "a later reply gives Juniper the digested deep read"


class _Bus:
    def __init__(self):
        self.sent = []

    async def publish(self, channel, envelope):
        self.sent.append((channel, envelope))

    def events(self):
        return [GrammarEventV1.model_validate(e.payload) for ch, e in self.sent if ch == "orion:grammar:event"]


def _item(quote, *, offer=False, force="commits", done_when=DONE):
    return {"quote": quote, "offer": offer, "doer": "orion", "force": force, "act": "x", "done_when": done_when}


def _reflection(**kw):
    base = dict(correlation_id="c2", thought_event_id="t", substrate_appraisal_id="s", draft_hash="h",
                imperative="i", tone="t", strain_refs=[], alignment_verdict="aligned", alignment_notes=[],
                strain_unresolved=False)
    base.update(kw)
    return FinalizeReflectionV1(**base)


@pytest.fixture(autouse=True)
def _flags_on(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HUB_PROMISE_LEDGER_ENABLED", True, raising=False)


def test_view_reads_quotes_from_the_turn_record_and_counts_unreadable_acts():
    ref0, ref9 = promise_ref("c1", 0), promise_ref("c1", 9)
    shown, unreadable = view_from_rows(
        [{"act_id": "act_a", "test_before": ref0}, {"act_id": "act_b", "test_before": ref9},
         {"act_id": "act_c", "test_before": "garbage"}],
        [{"correlation_id": "c1", "made": [_item("  " + QUOTE.replace(" ", "\n", 1))]}])
    assert shown == [OpenPromiseRefV1(id="act_a", quote=QUOTE)]
    assert unreadable == 2


def test_pending_offers_are_firm_offers_only():
    made = [_item(QUOTE), _item("If you want, I'll re-read it later.", offer=True),
            _item("Want me to dig into it tomorrow?", offer=True, done_when=None),
            {"quote": "an old offer from before force", "offer": True, "doer": "orion"}]
    assert offers_from_made("c1", made) == [(promise_ref("c1", 1), "If you want, I'll re-read it later.")]


@pytest.mark.asyncio
async def test_delivered_juniper_turn_opens_and_closes_promise_acts(monkeypatch):
    from orion.hub import turn_orchestrator
    from orion.grammar.promise_view import PromiseView

    offer_ref = promise_ref("c1", 1)

    view = PromiseView(offers=[(offer_ref, "If you want, I'll re-read it later.")])
    kept = promise_act_id(promise_ref("c0", 0))
    reflection = _reflection(
        promises_made=[PromiseV1(**_item(QUOTE)), PromiseV1(**_item("If you want, I'll look again.", offer=True))],
        offers_accepted=["If you want, I'll re-read it later."],
        promises_kept=[kept], promises_kept_evidence=[PromiseKeptV1(id=kept, quote="Here is the read.")])
    bus = _Bus()
    counts = await turn_orchestrator._publish_promise_acts(
        bus, correlation_id="c2", session_id="s1", source_label="hub_orion", client_meta=None, reflection=reflection,
        promise_view=view)
    kinds = [(e.event_kind, e.act.act_id) for e in bus.events()]
    assert kinds == [("act_opened", promise_act_id(promise_ref("c2", 0))),
                     ("act_opened", promise_act_id(offer_ref)),
                     ("act_closed", kept)]
    assert counts["opened"] == 1 and counts["opened_offers"] == 1 and counts["kept"] == 1
    assert counts["offers_not_opened"] == 1 and counts["published"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("source_label,client_meta,flag", [
    ("aitown", None, True),
    ("hub_orion", {"unsolicited": "true"}, True),
    ("hub_orion", None, False),
])
async def test_other_turns_and_flag_off_publish_no_promise_acts(monkeypatch, source_label, client_meta, flag):
    from scripts.settings import settings
    from orion.grammar.promise_view import PromiseView
    from orion.hub import turn_orchestrator

    monkeypatch.setattr(settings, "HUB_PROMISE_LEDGER_ENABLED", flag, raising=False)
    bus = _Bus()
    reflection = _reflection(promises_made=[PromiseV1(**_item(QUOTE))])
    assert await turn_orchestrator._publish_promise_acts(
        bus, correlation_id="c2", session_id="s1", source_label=source_label, client_meta=client_meta,
        reflection=reflection, promise_view=PromiseView()) is None
    assert bus.events() == []


@pytest.mark.asyncio
async def test_motor_prompt_never_shows_open_promises():
    """Acceptance check 5: the request carries open promises and pending offers for the judge; the FCC
    motor, driven through HarnessRunner with that request, never receives them in any kwarg."""
    from unittest.mock import AsyncMock

    from orion.harness.runner import HarnessRunner
    from orion.harness.tests.fixtures import make_thought

    secret, offer = "I'll bring the zebra-striped audit back on Thursday.", "If you want, I'll tour the okapi logs."
    req = HarnessRunRequestV1.model_validate({
        "correlation_id": "c-motor", "thought_event": make_thought().model_dump(mode="json"), "user_message": "hi",
        "permissions": {}, "answer_contract": {}, "situation_prompt_fragment": "Situation: quiet evening.",
        "open_promises": [{"id": "act_a", "quote": secret}], "pending_offers": [offer]})
    assert req.open_promises[0].quote == secret and req.pending_offers == [offer]
    captured: dict = {}

    async def _fcc(**kwargs):
        captured.update(kwargs)
        yield {"type": "final", "llm_response": "answer", "metadata": {"exit_code": 0}}

    await HarnessRunner(AsyncMock(), fcc_runner=_fcc, served_model_probe=AsyncMock(return_value=None),
                        pool_state_probe=AsyncMock(return_value=None)).run(req)
    assert "Situation: quiet evening." in captured["prompt"]  # the request's other fields did reach the motor
    blob = repr(captured)
    assert "zebra-striped" not in blob and "okapi" not in blob and "act_a" not in blob
    for rel in ("orion/harness/prefix.py", "orion/harness/runner.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "open_promises" not in src and "pending_offers" not in src, rel


@pytest.mark.asyncio
async def test_view_read_fails_open(monkeypatch):
    """A failed or slow read shows the judge nothing and never raises into the turn."""
    import time

    from orion.hub import promise_ledger

    monkeypatch.setattr(promise_ledger, "fetch_promise_view", lambda *a, **k: None)
    assert (await promise_ledger.promise_view_for_turn("s", "c")).open_promises == []

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(promise_ledger, "fetch_promise_view", _boom)
    assert (await promise_ledger.promise_view_for_turn("s", "c")).offers == []
    monkeypatch.setattr(promise_ledger, "fetch_promise_view", lambda *a, **k: time.sleep(0.5))
    started = time.monotonic()
    view = await promise_ledger.promise_view_for_turn("s", "c", timeout_sec=0.05)
    assert view.open_promises == [] and time.monotonic() - started < 0.4


@pytest.mark.asyncio
async def test_no_view_means_no_acts_even_on_a_hub_orion_turn():
    """Same rule as the judge's view: only a turn that got a pre-turn view (Juniper's own) opens acts."""
    from orion.hub import turn_orchestrator

    bus = _Bus()
    reflection = _reflection(promises_made=[PromiseV1(**_item(QUOTE))])
    assert await turn_orchestrator._publish_promise_acts(
        bus, correlation_id="c2", session_id="s1", source_label="hub_orion", client_meta=None,
        reflection=reflection, promise_view=None) is None
    assert bus.events() == []
