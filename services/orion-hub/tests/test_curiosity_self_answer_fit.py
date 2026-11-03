"""The fit hop: an over-budget self answer gets one more turn from the same
model to rewrite it into the reading budget, and the mirror holds that
rewrite. The full answer stays on the graph either way."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock

from orion.curiosity.self_inquiry import (
    SELF_DEFINITION_TEXT_CAP,
    SELF_FIT_DEFINITION_FIELD,
    SELF_FIT_TAG,
    SELF_INQUIRY_TAG,
    LivedAnswer,
)
from orion.schemas.durable_run import CuriosityTurnRequestV1
from orion.schemas.gpu_pool import GpuLeaseRefV1
from scripts import curiosity_investigation as ci

from test_curiosity_investigation import _FakeBus, _FakeReader, _graph_loop
from test_curiosity_self_inquiry import _GrantConn, _mirrors

RUN = "abcdef123456"
QID = "lived.care"
LONG = ("I looked at my own records and this is what held up under the check. " * 22).strip()
FIT = ("I checked my records; most of it held, and some of it I could not check. " * 6).strip()


class _LongAnswerReader(_FakeReader):
    """One LivedAnswer (and optionally a SelfDefinition) for any run id."""

    def __init__(self, *, text: str = LONG, definition_text: str | None = None) -> None:
        super().__init__()
        self.text = text
        self.definition_text = definition_text

    def query(self, cypher: str):
        self.queries.append(cypher)
        if "UNWIND" in cypher:
            return [{"e": "records:self_concept_history"}]
        m = re.search(r"\(s:SelfDefinition\) WHERE s\.run_id = '([0-9a-f]+)'", cypher)
        if m:
            if self.definition_text is None:
                return []
            return [{"run_id": m.group(1), "text": self.definition_text, "evidence": ["a"], "written_at": 1}]
        m = re.search(r"\(a:LivedAnswer\) WHERE a\.run_id = '([0-9a-f]+)'", cypher)
        if m:
            return [{"run_id": m.group(1), "question_id": QID, "family": "lived", "text": self.text,
                     "evidence": ["records:self_concept_history"], "revises": "", "written_at": 1}]
        return []


def _loop(reader):
    bus = _FakeBus()
    return bus, _graph_loop(bus, reader=reader, conn=_GrantConn(), self_inquiry_enabled=True)


def _with_generate(loop, reply=FIT):
    calls = []

    async def generate(prompt, corr, **kwargs):
        calls.append((prompt, corr, kwargs))
        return reply, {}

    loop._generate = generate
    return calls


def test_an_over_budget_answer_gets_one_fit_hop_and_the_mirror_holds_it() -> None:
    bus, loop = _loop(_LongAnswerReader())
    calls = _with_generate(loop)
    assert asyncio.run(loop._fit_run_answers(RUN, "corr-1")) == 1
    assert len(calls) == 1
    prompt, _corr, kwargs = calls[0]
    assert LONG in prompt
    assert kwargs["source"] == SELF_FIT_TAG
    assert kwargs["require_lookup"] is False
    assert kwargs["resume_key"] == f"{RUN}:{SELF_FIT_TAG}:{QID}"
    assert bus.redis.hashes[f"orion:curiosity:self_fit:{RUN}"][QID] == FIT

    answer = LivedAnswer(RUN, QID, "lived", LONG, ["records:self_concept_history"])
    assert asyncio.run(loop._mirror_lived_answer(answer, run_id=RUN, correlation_id="corr-1"))
    (_, env), = _mirrors(bus)
    assert env.payload["content"] == FIT


def test_a_second_call_for_the_same_run_does_not_refit() -> None:
    _bus, loop = _loop(_LongAnswerReader())
    calls = _with_generate(loop)
    asyncio.run(loop._fit_run_answers(RUN, "corr-1"))
    asyncio.run(loop._fit_run_answers(RUN, "corr-1"))
    assert len(calls) == 1


def test_an_answer_inside_the_budget_gets_no_hop() -> None:
    _bus, loop = _loop(_LongAnswerReader(text="I care about continuity with Juniper."))
    calls = _with_generate(loop)
    assert asyncio.run(loop._fit_run_answers(RUN, "corr-1")) == 0
    assert calls == []


def test_a_long_definition_is_fitted_under_its_own_field() -> None:
    bus, loop = _loop(_LongAnswerReader(text="short.", definition_text=LONG))
    _with_generate(loop)
    assert asyncio.run(loop._fit_run_answers(RUN, "corr-1")) == 1
    assert bus.redis.hashes[f"orion:curiosity:self_fit:{RUN}"][SELF_FIT_DEFINITION_FIELD] == FIT


def test_a_refused_reply_leaves_the_full_text_as_the_mirror() -> None:
    bus, loop = _loop(_LongAnswerReader())
    _with_generate(loop, reply="Done.")
    assert asyncio.run(loop._fit_run_answers(RUN, "corr-1")) == 0
    answer = LivedAnswer(RUN, QID, "lived", LONG, ["records:self_concept_history"])
    asyncio.run(loop._mirror_lived_answer(answer, run_id=RUN, correlation_id="corr-1"))
    (_, env), = _mirrors(bus)
    assert env.payload["content"] == LONG  # every word kept; readers fit at a sentence


def test_under_a_gpu_hold_the_hop_carries_the_hold_and_the_finalize_reserve() -> None:
    _bus, loop = _loop(_LongAnswerReader())
    loop.held_turn_finalize_reserve_sec = 330.0
    calls = _with_generate(loop)
    lease = GpuLeaseRefV1(lease_id="hold-1", generation=1, role="agent", holder=f"durable-runs:{RUN}")
    deadline = ci.time.monotonic() + 10_000
    assert asyncio.run(loop._fit_run_answers(RUN, "corr-1", gpu_lease=lease, deadline=deadline)) == 1
    kwargs = calls[0][2]
    assert kwargs["gpu_lease"] == lease
    assert kwargs["fcc_model_label"].endswith(ci.GPU_LEASE_ROUTE)
    assert kwargs["timeout_sec"] == ci.SELF_FIT_TIMEOUT_SEC + 330.0


def test_no_hop_when_the_hold_is_about_to_end() -> None:
    bus, loop = _loop(_LongAnswerReader())
    loop.held_turn_finalize_reserve_sec = 330.0
    calls = _with_generate(loop)
    lease = GpuLeaseRefV1(lease_id="hold-1", generation=1, role="agent", holder=f"durable-runs:{RUN}")
    deadline = ci.time.monotonic() + 330.0 + 40.0  # motor would get ~10 s
    assert asyncio.run(loop._fit_run_answers(RUN, "corr-1", gpu_lease=lease, deadline=deadline)) == 0
    assert calls == []
    assert f"orion:curiosity:self_fit:{RUN}" not in bus.redis.hashes


def test_a_self_inquiry_turn_runs_the_fit_hop_after_the_answer(monkeypatch) -> None:
    monkeypatch.setattr(ci, "validate_hold_ref", AsyncMock())
    bus, loop = _loop(_LongAnswerReader())
    calls = _with_generate(loop, reply=FIT)
    request = CuriosityTurnRequestV1(
        run_id=RUN, correlation_id="corr-1", prompt="Self-inquiry", timeout_sec=5000,
        source_tag=SELF_INQUIRY_TAG,
    )
    result = asyncio.run(loop._turn_result_for(request, hold_lock=False))
    assert result.ok
    sources = [kw["source"] for _p, _c, kw in calls]
    assert sources == [SELF_INQUIRY_TAG, SELF_FIT_TAG]
    assert bus.redis.hashes[f"orion:curiosity:self_fit:{RUN}"][QID] == FIT


def test_the_cap_constant_is_still_the_budget_orion_is_told() -> None:
    assert SELF_DEFINITION_TEXT_CAP == 1200


def test_the_live_eval_flags_only_a_mid_word_cut_at_the_cap() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "evals" / "run_self_answer_fit_eval.py"
    spec = importlib.util.spec_from_file_location("run_self_answer_fit_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.classify(1200, "r") == "at_cap_mid_word"
    assert mod.classify(1200, ".") == "within_budget"
    assert mod.classify(1180, "x") == "within_budget"
    assert mod.classify(1398, ".") == "full_over_budget"
