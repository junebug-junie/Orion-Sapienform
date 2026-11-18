"""A sleep-started dream.trigger keeps its sleep digest all the way into the
dream_cycle prompt: dispatch_dream_trigger -> cortex.orch.request ->
build_plan_request (plan + context) -> the template exec renders."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, APP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.orchestrator import build_plan_request, dispatch_dream_trigger  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.cortex.contracts import CortexClientRequest  # noqa: E402
from orion.schemas.telemetry.dream import DreamInternalTriggerV1, DreamSleepDigestV1  # noqa: E402

SRC = ServiceRef(name="test", version="0", node="n")


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, env):
        self.published.append((channel, env))


def _dispatch(payload):
    bus = _Bus()
    asyncio.run(dispatch_dream_trigger(bus, source=SRC, env=BaseEnvelope(kind="dream.trigger", source=SRC, payload=payload)))
    assert len(bus.published) == 1
    return CortexClientRequest.model_validate(bus.published[0][1].payload)


def _rendered_prompt(req):
    plan_req = build_plan_request(req, "corr-1")
    template = next(s.prompt_template for s in plan_req.plan.steps if s.prompt_template)
    return Environment(autoescape=False).from_string(template).render(
        memory_digest="(memories)", **plan_req.context)


def test_sleep_digest_reaches_the_rendered_dream_prompt():
    trigger = DreamInternalTriggerV1(
        trigger_id="sleep:dc-1", source="orion-dream.sleep",
        sleep=DreamSleepDigestV1(cycle_id="dc-1", started_at="2026-10-09T06:33:00Z", pressure=13.26, threshold=3.0,
                                 material=["metacog: gateway saturated", "resonance: ring quiet"]),
    )
    req = _dispatch(trigger.model_dump(mode="json"))
    assert req.verb == "dream_cycle"
    assert req.context.metadata["dream_trigger"]["sleep"]["material"] == ["metacog: gateway saturated", "resonance: ring quiet"]
    prompt = _rendered_prompt(req)
    assert "TONIGHT'S SLEEP" in prompt and "sleep dc-1" in prompt and "against a sleep line of 3.0" in prompt
    assert "- metacog: gateway saturated" in prompt


def test_hand_started_dream_still_gets_the_memory_only_prompt():
    prompt = _rendered_prompt(_dispatch({"mode": "standard"}))
    assert "TONIGHT'S SLEEP" not in prompt
    assert "Synthesize a dream narrative from the memory bundle and mode." in prompt
