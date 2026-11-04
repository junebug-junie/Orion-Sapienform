"""A tiny second loop instance for tests: proves the runtime is not Prime-shaped.

``EchoRuntime`` thinks once per queued request (like the curiosity replica's one model call per plan
request), bids family ``test.echo`` at tier 2 on lane chat, and keeps its own state under its own Redis key.
Fakes for the bus, model and market are here too, so orion-mind's tests can run it next to Prime.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.grammar.acts import build_act
from orion.mindloop.backoff import TryKind
from orion.mindloop.instance import LoopIdentity
from orion.mindloop.runtime import MindLoopRuntime, ModelReply
from orion.mindloop.turns import JuniperTurns

T0 = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
ECHO_KEY = "orion:test_echo:state"

ECHO_IDENTITY = LoopIdentity(
    name="echo", family="test.echo", actor="echo", producer="test.echo", state_key=ECHO_KEY,
    settings_prefix="ECHO_", source_service="orion-mind", thought_component="echo.thought", thought_ref="echo_thought:",
    tier=2, logger_name="test.mindloop.echo",
)


class EchoStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    told: list[str] = Field(default_factory=list)
    last_thought_at: Optional[AwareDatetime] = None
    no_model_streak: int = 0
    no_model_tries_total: int = 0
    last_try_kind: Optional[TryKind] = None
    retry_not_before: Optional[AwareDatetime] = None


class EchoRuntime(MindLoopRuntime):
    identity = ECHO_IDENTITY
    state_model = EchoStateV1

    async def _upkeep(self) -> tuple[Any, datetime]:
        return None, self.clock()

    def _wants_to_think(self, now: datetime) -> bool:
        if not self.state.requests:
            return False
        return self.state.retry_not_before is None or now >= self.state.retry_not_before

    def _pending_count(self) -> int:
        return len(self.state.requests)

    def _tell(self, text: str) -> None:
        self.state = self.state.model_copy(update={"told": [*self.state.told, text]})

    def _thought_act(self, thought_id: str, horizon_sec: float):
        return build_act(actor=self.identity.actor, verb="infer", key=thought_id, signal_id="artifact_produced",
                         horizon_sec=horizon_sec, basis="test", object_ref=f"echo_thought:{thought_id}")

    def _woke_note(self) -> str:
        return f"woke: {len(self.state.requests)} request(s)"

    async def _ask(self, inputs: Any, now: datetime) -> ModelReply:
        return await self._ask_model("echo the request", self.state.requests[0])

    def _parse_answer(self, parsed: Optional[dict[str, Any]]) -> tuple[Any, Optional[str]]:
        text = (parsed or {}).get("echo")
        return (text, None) if isinstance(text, str) and text else (None, "no echo")

    async def _conclude(self, act, thought_id, inputs, now, reply, answer) -> tuple[bool, str]:
        self.state = self.state.model_copy(update={"requests": self.state.requests[1:],
                                                   "answers": [*self.state.answers, answer]})
        return True, f"echoed {answer!r}"


class Redis:
    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value):
        self.kv[key] = value


class Bus:
    def __init__(self) -> None:
        self.redis = Redis()
        self.published: list = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))

    def acts(self):
        return [(e.payload["event_kind"], e.payload["act"]) for c, e in self.published if c == "orion:grammar:event"]


class SeqLLM:
    def __init__(self, results) -> None:
        self.results, self.calls = list(results), []

    def request_json(self, **kw):
        self.calls.append(kw)
        return self.results.pop(0)


SETTINGS = SimpleNamespace(
    SERVICE_NAME="test", SERVICE_VERSION="t", NODE_NAME="test", ECHO_TICK_SEC=15, ECHO_REFRACTORY_SEC=0,
    ECHO_BID_LANE="chat", ECHO_LLM_ROUTE="echo", ECHO_STARVED_LLM_ROUTE="echo_starved", ECHO_SEAT_WAIT_SEC=90,
    ECHO_LLM_TIMEOUT_SEC=30, ECHO_MAX_TOKENS=200,
)


def echo_runtime(results, *, bus=None, outcome="funded", turns=None, requests=("hello",)) -> EchoRuntime:
    clock = {"now": T0}
    bids, released = [], []

    async def bid(want):
        bids.append(want)
        return SimpleNamespace(outcome=outcome, reason="test", binding_constraint=None)

    async def release(want_id):
        released.append(want_id)

    rt = EchoRuntime(settings=SimpleNamespace(**vars(SETTINGS)), bus=bus or Bus(), llm=SeqLLM(results), bid=bid,
                     release=release, clock=lambda: clock["now"], turns=turns or JuniperTurns())
    rt.state = rt.state.model_copy(update={"requests": list(requests)})
    rt.bids, rt.released, rt.clock_box = bids, released, clock
    return rt


def advance(rt: EchoRuntime, sec: float) -> None:
    rt.clock_box["now"] = rt.clock_box["now"] + timedelta(seconds=sec)
