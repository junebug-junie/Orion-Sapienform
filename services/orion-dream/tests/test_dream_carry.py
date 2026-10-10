"""dream.carry, orion-dream's half: text-hop prompts/parsing/handler, the finished dream,
the step listener, and the submit to cortex-orch."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from orion.schemas.dream_carry import (
    DREAM_CARRY_LLM_ROUTE,
    IMAGE_PROMPT_MAX_WORDS,
    DREAM_CARRY_STEP_REPLY_PREFIX,
    DREAM_CARRY_STEP_REQUEST_KIND,
    DREAM_CARRY_STEP_RESULT_KIND,
    DreamCarryBriefV1,
    DreamCarryHopV1,
    DreamCarryStepRequestV1,
    dream_carry_run_id,
)
from orion.schemas.telemetry.dream import DreamSleepDigestV1

LEASE = {"lease_id": "lease-1", "generation": 3, "role": "metacog_background", "holder": "durable:dream-carry-x"}
SHA = "a" * 64
MATERIAL = [
    "metacog: rpc timeout to the vision host recurred four times",
    "crystallization: Juniper said the porch light is on a timer",
    "reverie: a compaction ask about old chat summaries",
]
HYPOTHESIS = "These two recur together more often than chance would allow"


def _sleep(**kw):
    return DreamSleepDigestV1(cycle_id="dc-abc123", started_at=datetime(2026, 10, 9, 6, 33, tzinfo=timezone.utc),
                              pressure=13.26, threshold=3.0, material=MATERIAL, **kw)


def _brief(sleep="default"):
    return DreamCarryBriefV1(trigger_id="sleep:dc-abc123", sleep=_sleep() if sleep == "default" else sleep)


def _text(i, passage=None, prompt="a lamp on a porch at dusk"):
    return DreamCarryHopV1(kind="text", index=i, passage=passage or f"passage {i}. it goes on.", image_prompt=prompt)


def _image(i, caption=None):
    return DreamCarryHopV1(kind="image", index=i, sha256=SHA, caption=caption or f"caption {i}", child_run_id=f"child-{i}")


def _req(step="text", hop_index=0, hops=(), brief=None, stopped_reason=None, run_id="dream-carry-run1"):
    return DreamCarryStepRequestV1(
        run_id=run_id, correlation_id="11111111-1111-1111-1111-111111111111", step=step,
        brief=brief or _brief(), hops=list(hops), hop_index=hop_index if step == "text" else None,
        gpu_lease=LEASE if step == "text" else None, stopped_reason=stopped_reason,
    )


# --- prompts ---------------------------------------------------------------------


def test_first_prompt_carries_the_sleep_material_and_never_a_hypothesis():
    from app.carry import text_prompt

    prompt = text_prompt(_req())
    for item in MATERIAL:
        assert f"- {item}" in prompt
    assert "tiredness 13.26 against a sleep line of 3.0" in prompt and "dc-abc123" in prompt
    assert "overdue" not in prompt
    assert HYPOTHESIS not in prompt and "hypothes" not in prompt.lower()
    assert '"passage"' in prompt and '"image_prompt"' in prompt and "45 words" in prompt
    assert "no text, letters" in prompt


def test_first_prompt_says_when_the_sleep_was_overdue():
    from app.carry import text_prompt

    assert "overdue, slept on the 48 h backstop" in text_prompt(_req(brief=_brief(_sleep(overdue=True))))


def test_hand_started_prompt_admits_it_has_no_material():
    from app.carry import text_prompt

    prompt = text_prompt(_req(brief=DreamCarryBriefV1(trigger_id="manual:x")))
    assert "Dream freely from your recent days" in prompt and "TONIGHT'S SLEEP" not in prompt


def test_continue_prompt_carries_the_last_caption_and_the_previous_passage():
    from app.carry import text_prompt

    hops = [_text(0, "The porch light hummed like a question."), _image(1, "two light bulbs hanging from wires")]
    prompt = text_prompt(_req(hop_index=2, hops=hops))
    assert "The porch light hummed like a question." in prompt
    assert "looking at it you see: two light bulbs hanging from wires" in prompt
    assert "follow the picture" in prompt
    # hop 4 continues from hop 2's passage and hop 3's caption, not the first pair
    hops4 = hops + [_text(2, "Second passage here."), _image(3, "a desk with a laptop")]
    p4 = text_prompt(_req(hop_index=4, hops=hops4))
    assert "Second passage here." in p4 and "a desk with a laptop" in p4
    assert "The porch light hummed" not in p4 and "two light bulbs" not in p4


# --- parsing / text handler ---------------------------------------------------------------


def _complete(reply, calls):
    async def complete(prompt, gpu_lease, timeout_sec):
        calls.append({"prompt": prompt, "gpu_lease": gpu_lease, "timeout": timeout_sec})
        if isinstance(reply, Exception):
            raise reply
        return reply
    return complete


def test_good_reply_is_a_done_hop_with_the_prompt_clipped_to_60_words():
    from app.carry import handle_text

    long_prompt = " ".join(f"w{i}" for i in range(75))
    calls = []
    reply = json.dumps({"passage": "  A long corridor of lamps.  ", "image_prompt": long_prompt})
    result = asyncio.run(handle_text(_req(), _complete(reply, calls)))
    assert result.status == "done" and result.hop.kind == "text" and result.hop.index == 0
    assert result.hop.passage == "A long corridor of lamps."
    assert result.hop.image_prompt.split() == [f"w{i}" for i in range(IMAGE_PROMPT_MAX_WORDS)]
    assert calls[0]["gpu_lease"] == LEASE  # the run's hold, forwarded
    assert result.run_id == "dream-carry-run1" and result.step == "text"


def test_fenced_or_wrapped_json_is_accepted():
    from app.carry import parse_text_reply

    body = json.dumps({"passage": "p", "image_prompt": "a red door"})
    assert parse_text_reply(f"```json\n{body}\n```") == ("p", "a red door")
    assert parse_text_reply(f"Here is the dream:\n{body}") == ("p", "a red door")


@pytest.mark.parametrize("reply", [
    "", "not json at all", json.dumps({"passage": "", "image_prompt": "x"}),
    json.dumps({"passage": "p", "image_prompt": "   "}), json.dumps(["p", "x"]),
    json.dumps({"passage": "p"}),
])
def test_empty_or_unparseable_reply_retries_never_a_blank_hop(reply):
    from app.carry import handle_text

    result = asyncio.run(handle_text(_req(), _complete(reply, [])))
    assert result.status == "retry" and result.hop is None and result.reason.startswith("reply_")
    assert result.retry_after_sec and result.retry_after_sec > 0


@pytest.mark.parametrize("exc_name", ["GatewayRefused", "TimeoutError"])
def test_refused_or_transport_failure_retries(exc_name):
    from app import llm
    from app.carry import handle_text

    exc = llm.GatewayRefused("pool_shed:busy") if exc_name == "GatewayRefused" else TimeoutError("rpc")
    result = asyncio.run(handle_text(_req(), _complete(exc, [])))
    assert result.status == "retry" and exc_name in result.reason


def test_a_continue_hop_without_its_prior_hops_is_terminal():
    from app.carry import handle_text

    calls = []
    result = asyncio.run(handle_text(_req(hop_index=2, hops=[_text(0)]), _complete("{}", calls)))
    assert result.status == "terminal" and "missing_prior_hops" in result.reason and calls == []


def test_llm_complete_forwards_gpu_lease_route_and_tokens():
    """The real gateway call: options carry the hold, the carry route and the passage budget."""
    from app import llm
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.bus.codec import OrionCodec

    sent = {}

    class Bus:
        codec = OrionCodec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
            sent.update(channel=channel, env=env, timeout=timeout_sec)
            return {"data": self.codec.encode(BaseEnvelope(kind="llm.chat.result", source=ServiceRef(name="gw"),
                                                           payload={"content": "{\"passage\": \"p\"}"}))}

    text = asyncio.run(llm.complete(Bus(), "hi", max_tokens=700, purpose="dream_carry",
                                    route=DREAM_CARRY_LLM_ROUTE, gpu_lease=LEASE, timeout_sec=120.0))
    assert text.startswith("{")
    opts = sent["env"].payload["options"]
    assert opts["gpu_lease"] == LEASE and opts["max_tokens"] == 700 and opts["purpose"] == "dream_carry"
    assert opts["llm_route"] == DREAM_CARRY_LLM_ROUTE and sent["env"].payload["route"] == DREAM_CARRY_LLM_ROUTE
    assert sent["timeout"] == 120.0 and opts["gateway_read_timeout_sec"] == 120.0
    # the recombination call is unchanged: no lease, 320 tokens
    asyncio.run(llm.complete(Bus(), "hi"))
    opts = sent["env"].payload["options"]
    assert "gpu_lease" not in opts and opts["max_tokens"] == 320 and opts["purpose"] == "dream_recombine"


# --- finish ---------------------------------------------------------------------------


def _six():
    return [_text(0, "First. Then more."), _image(1, "two bulbs"), _text(2, "Second."),
            _image(3, "a desk"), _text(4, "Third."), _image(5, "a porch at sunset")]


def test_finished_dream_keeps_hops_in_order_and_interleaves_captions():
    from app.carry import build_carry_dream, carry_dream_id

    hops = _six()
    dream = build_carry_dream(_req("finish", hops=list(reversed(hops))))
    assert [f["id"] for f in dream.fragments] == [f"hop-{i}" for i in range(6)]
    assert [f["kind"] for f in dream.fragments] == ["text", "image"] * 3
    assert dream.fragments[1] == {"id": "hop-1", "kind": "image", "index": 1, "sha256": SHA,
                                  "caption": "two bulbs", "child_run_id": "child-1"}
    assert dream.fragments[0]["passage"] == "First. Then more." and dream.fragments[0]["image_prompt"]
    assert dream.narrative == ("First. Then more.\n\n[picture] two bulbs\n\nSecond.\n\n[picture] a desk"
                               "\n\nThird.\n\n[picture] a porch at sunset")
    assert dream.mode == "carry" and dream.profile == "dream.carry" and dream.themes == []
    assert dream.tldr.startswith("First.") and len(dream.tldr) < 400
    assert dream.trigger["trigger_id"] == "sleep:dc-abc123" and dream.trigger["sleep"]["cycle_id"] == "dc-abc123"
    assert dream.trigger["carry_run_id"] == "dream-carry-run1" and dream.trigger["stopped_reason"] is None
    assert dream.source_context == {"carry_run_id": "dream-carry-run1", "hops_made": 6, "stopped_reason": None}
    assert dream.dream_id == carry_dream_id("dream-carry-run1") == build_carry_dream(_req("finish", hops=hops)).dream_id
    assert dream.dream_id != carry_dream_id("dream-carry-run2")
    # sql-writer's projection keeps the trigger link in metrics._dream_audit
    assert dream.merged_metrics_for_sql()["_dream_audit"]["dream_id"] == dream.dream_id


def test_partial_carry_publishes_what_it_made_with_the_stopped_reason():
    from app.carry import build_carry_dream

    dream = build_carry_dream(_req("finish", hops=_six()[:3], stopped_reason="thermal_refused at hop 3"))
    assert len(dream.fragments) == 3 and dream.narrative.endswith("Second.")
    assert dream.trigger["stopped_reason"] == "thermal_refused at hop 3"
    assert dream.source_context["stopped_reason"] == "thermal_refused at hop 3"


def test_zero_hops_is_terminal_and_publishes_nothing():
    from app.carry import FinishLedger, handle_finish

    published = []

    async def publish(d):
        published.append(d)

    result = asyncio.run(handle_finish(_req("finish", stopped_reason="deadline at hop 0"), publish, FinishLedger()))
    assert result.status == "terminal" and result.reason == "no_hops" and published == []


def test_a_replayed_finish_publishes_once():
    from app.carry import FinishLedger, handle_finish

    published, ledger = [], FinishLedger()

    async def publish(d):
        published.append(d)

    first = asyncio.run(handle_finish(_req("finish", hops=_six()), publish, ledger))
    again = asyncio.run(handle_finish(_req("finish", hops=_six()), publish, ledger))
    assert first.status == again.status == "done" and first.dream_id == again.dream_id
    assert len(published) == 1


def test_a_finish_already_in_dreams_is_not_republished_after_a_restart():
    from app.carry import FinishLedger, carry_dream_id, handle_finish

    published, asked = [], []

    async def publish(d):
        published.append(d)

    async def recorded(dream_id):
        asked.append(dream_id)
        return True

    result = asyncio.run(handle_finish(_req("finish", hops=_six()), publish, FinishLedger(), recorded))
    assert result.status == "done" and published == [] and asked == [carry_dream_id("dream-carry-run1")]


def test_a_failed_row_check_still_publishes_and_a_failed_publish_retries():
    from app.carry import FinishLedger, handle_finish

    async def broken_check(_):
        raise RuntimeError("db down")

    published = []

    async def publish(d):
        published.append(d)

    assert asyncio.run(handle_finish(_req("finish", hops=_six()), publish, FinishLedger(), broken_check)).status == "done"
    assert len(published) == 1

    async def broken_publish(_):
        raise ConnectionError("bus down")

    ledger = FinishLedger()
    result = asyncio.run(handle_finish(_req("finish", hops=_six()), broken_publish, ledger))
    assert result.status == "retry" and ledger.get("dream-carry-run1") is None


# --- listener -------------------------------------------------------------------------


class _Bus:
    def __init__(self, reply="{}"):
        from orion.core.bus.codec import OrionCodec

        self.codec = OrionCodec()
        self.published = []
        self.reply = reply

    async def publish(self, channel, env):
        self.published.append((channel, env))


def _listener(bus, reply):
    from app.carry_listener import DreamCarryListener
    from orion.core.bus.bus_schemas import ServiceRef

    lst = DreamCarryListener(bus_url="redis://x", source=ServiceRef(name="orion-dream"), dream_log_channel="orion:dream:log")
    lst.bus = bus

    async def complete(prompt, gpu_lease, timeout_sec):
        return reply
    lst.complete = complete
    return lst


def _env(request, reply_to=None):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    return BaseEnvelope(kind=DREAM_CARRY_STEP_REQUEST_KIND, source=ServiceRef(name="orion-durable-runs"),
                        correlation_id=request.correlation_id,
                        reply_to=reply_to or f"{DREAM_CARRY_STEP_REPLY_PREFIX}:{request.correlation_id}",
                        payload=request.model_dump(mode="json"))


def test_listener_answers_a_text_step_on_reply_to_with_echoed_identity():
    bus = _Bus()
    lst = _listener(bus, json.dumps({"passage": "p", "image_prompt": "a red door"}))
    request = _req()
    env = _env(request)
    asyncio.run(lst.handle(env))
    (channel, reply), = bus.published
    assert channel == env.reply_to and reply.kind == DREAM_CARRY_STEP_RESULT_KIND
    assert reply.correlation_id == env.correlation_id
    assert reply.payload["run_id"] == request.run_id and reply.payload["step"] == "text"
    assert reply.payload["status"] == "done" and reply.payload["hop"]["image_prompt"] == "a red door"


def test_listener_publishes_the_finished_dream_to_the_dream_log_then_replies():
    bus = _Bus()
    lst = _listener(bus, "")
    asyncio.run(lst.handle(_env(_req("finish", hops=_six()))))
    (log_channel, dream_env), (reply_channel, reply) = bus.published
    assert log_channel == "orion:dream:log" and dream_env.kind == "dream.result.v1"
    assert dream_env.payload["dream_id"] == reply.payload["dream_id"] and reply.payload["status"] == "done"
    assert reply_channel.startswith(DREAM_CARRY_STEP_REPLY_PREFIX)


def test_listener_ignores_foreign_reply_channels_and_kinds():
    bus = _Bus()
    lst = _listener(bus, "")
    assert asyncio.run(lst.handle(_env(_req(), reply_to="orion:hub:somewhere"))) is None
    env = _env(_req()).model_copy(update={"kind": "something.else"})
    assert asyncio.run(lst.handle(env)) is None
    assert bus.published == []


def test_listener_answers_an_invalid_request_terminal_when_it_can_name_it():
    bus = _Bus()
    lst = _listener(bus, "")
    env = _env(_req())
    env = env.model_copy(update={"payload": {**env.payload, "gpu_lease": None}})  # a text step without the hold
    asyncio.run(lst.handle(env))
    (_, reply), = bus.published
    assert reply.payload["status"] == "terminal" and reply.payload["reason"].startswith("invalid_request")


# --- submit -----------------------------------------------------------------------------


def test_carry_request_is_the_dream_carry_workflow_with_background_admission():
    from app.carry_submit import build_carry_request

    now = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)
    req = build_carry_request("sleep:dc-abc123", _sleep(), deadline_sec=14400, now=now)
    assert req.workflow == "dream.carry" and req.run_id == dream_carry_run_id("sleep:dc-abc123")
    assert req.brief.sleep == _sleep() and req.brief.trigger_id == "sleep:dc-abc123"
    assert req.admission.resource == "llm.route.metacog_background" and req.admission.priority == "background"
    assert (req.admission.deadline_at - now).total_seconds() == 14400
    assert build_carry_request("sleep:dc-abc123", _sleep(), deadline_sec=1).correlation_id == req.correlation_id


def _cortex_bus(receipt=None, status="accepted"):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.bus.codec import OrionCodec

    class Bus:
        codec = OrionCodec()

        def __init__(self):
            self.calls = []

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
            self.calls.append((channel, env))
            durable = env.payload["context"]["metadata"]["durable_run"]
            r = receipt if receipt is not None else {"run_id": durable["run_id"], "workflow_kind": "dream.carry",
                                                     "requested_resource": "llm.route.metacog_background"}
            return {"data": self.codec.encode(BaseEnvelope(
                kind="cortex.orch.result", source=ServiceRef(name="orion-cortex-orch"),
                payload={"status": status, "metadata": {"durable_run": r}}))}
    return Bus()


def test_submit_verifies_the_receipt_names_this_run():
    from app.carry_submit import build_carry_request, submit_via_cortex
    from orion.core.bus.bus_schemas import ServiceRef

    req = build_carry_request("sleep:dc-abc123", _sleep(), deadline_sec=60)
    bus = _cortex_bus()
    assert asyncio.run(submit_via_cortex(bus=bus, source=ServiceRef(name="orion-dream"), request=req,
                                         request_channel="orion:cortex:request")) is None
    channel, env = bus.calls[0]
    assert channel == "orion:cortex:request" and env.reply_to.startswith("orion:cortex:result:")
    assert env.payload["context"]["metadata"]["durable_run"]["brief"]["sleep"]["material"] == MATERIAL
    wrong = _cortex_bus(receipt={"run_id": "other", "workflow_kind": "dream.carry"})
    assert asyncio.run(submit_via_cortex(bus=wrong, source=ServiceRef(name="orion-dream"), request=req,
                                         request_channel="c")) == "receipt_run_id_mismatch"
    refused = _cortex_bus(status="fail")
    assert asyncio.run(submit_via_cortex(bus=refused, source=ServiceRef(name="orion-dream"), request=req,
                                         request_channel="c")).startswith("not_accepted:fail")


def test_a_hop_that_keeps_answering_badly_turns_terminal_after_the_cap():
    from app.carry import MAX_REPLY_FAILURES, ReplyFailures, handle_text

    failures, calls = ReplyFailures(), []
    statuses = [asyncio.run(handle_text(_req(), _complete("not json", calls), failures=failures)).status
                for _ in range(MAX_REPLY_FAILURES)]
    assert statuses == ["retry"] * (MAX_REPLY_FAILURES - 1) + ["terminal"]
    # a different hop (or run) has its own count; a good reply clears it
    assert asyncio.run(handle_text(_req(run_id="dream-carry-run2"), _complete("x", []), failures=failures)).status == "retry"
    good = json.dumps({"passage": "p", "image_prompt": "a door"})
    assert asyncio.run(handle_text(_req(run_id="dream-carry-run2"), _complete(good, []), failures=failures)).status == "done"
    assert asyncio.run(handle_text(_req(run_id="dream-carry-run2"), _complete("x", []), failures=failures)).status == "retry"


def test_refusals_never_turn_terminal():
    from app import llm
    from app.carry import MAX_REPLY_FAILURES, ReplyFailures, handle_text

    failures = ReplyFailures()
    for _ in range(MAX_REPLY_FAILURES + 2):
        r = asyncio.run(handle_text(_req(), _complete(llm.GatewayRefused("busy:"), []), failures=failures))
        assert r.status == "retry"


def test_a_text_step_queued_past_its_budget_retries_without_calling_the_llm():
    from app.carry import handle_text

    calls = []
    result = asyncio.run(handle_text(_req(), _complete("{}", calls), waited_sec=170.0))
    assert result.status == "retry" and "queued_past_budget" in result.reason and calls == []
    asyncio.run(handle_text(_req(), _complete(json.dumps({"passage": "p", "image_prompt": "d"}), calls), waited_sec=20.0))
    assert calls[0]["timeout"] == 180.0 - 15.0 - 20.0


def test_finished_dream_is_always_timestamped():
    from app.carry import build_carry_dream

    dream = build_carry_dream(_req("finish", hops=_six()))
    assert dream.created_at is not None and dream.created_at.tzinfo is not None
    assert dream.dream_date == dream.created_at.date()


def test_a_listener_handler_bug_retries_then_turns_terminal():
    from app import carry as carry_mod

    bus = _Bus()
    lst = _listener(bus, "")

    async def boom(*a, **kw):
        raise KeyError("bug")

    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(carry_mod, "handle_step", boom)
    try:
        for _ in range(carry_mod.MAX_REPLY_FAILURES):
            asyncio.run(lst.handle(_env(_req())))
    finally:
        mp.undo()
    statuses = [env.payload["status"] for _, env in bus.published]
    assert statuses == ["retry"] * (carry_mod.MAX_REPLY_FAILURES - 1) + ["terminal"]
