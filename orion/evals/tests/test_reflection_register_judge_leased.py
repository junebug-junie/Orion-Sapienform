"""The reflection-register judge (scripts/analysis/reflection_register_sources.judge_one) and its three
callers call only through an operator lane (orion/evals/leased_llm.py), never a seat URL (#3921: the door
replay sent 286+ unleased calls to the agent seat on 2026-11-25), and a reply that echoes the operator
route reads the served model from the grant. No pool, model or network here."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from orion.evals.leased_llm import LaneCallError, LaneRecalled

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "scripts/analysis"
SOURCES = ANALYSIS / "reflection_register_sources.py"
SHARE = ANALYSIS / "eval_reflection_register_share.py"
ANCHOR = ANALYSIS / "judge_anchor_set.py"
REPLAY = ROOT / "services/orion-hub/evals/run_self_question_door_replay.py"
PASSED = "qwen3.8-flash-next-iq3_s"


def _load(path: Path, name: str, monkeypatch):
    monkeypatch.syspath_prepend(str(ANALYSIS))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def src(monkeypatch):
    return _load(SOURCES, "reflection_register_sources", monkeypatch)


def _reply(register: str = "operational", model: str = PASSED) -> dict:
    return {"model": model, "choices": [{"message": {"content": json.dumps({"register": register, "reason": "r"})}}]}


class FakeLane:
    def __init__(self, reply=None, error=None, recall_after: int | None = None):
        self.bodies: list[dict] = []
        self.timeouts: list = []
        self.reply, self.error, self.recall_after = reply, error, recall_after

    def chat(self, body, timeout_sec=None):
        if self.recall_after is not None and len(self.bodies) >= self.recall_after:
            raise LaneRecalled("seat taken")
        self.bodies.append(body)
        self.timeouts.append(timeout_sec)
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.mark.parametrize("path", [SOURCES, SHARE, ANCHOR, REPLAY])
def test_no_direct_url_mode_and_no_worker_address(path: Path) -> None:
    src = path.read_text()
    assert "httpx" not in src and "urllib.request" not in src
    assert "/chat/completions" not in src
    assert not re.search(r"https?://[\w.\-<>]+:\d+", src)
    assert "judge-url" not in src and "judge_url" not in src


def test_judge_labels_through_the_lane_with_pinned_sampling(src) -> None:
    lane = FakeLane(_reply("philosophical"))
    item = src.Item("outreach", "outreach:1", "q", "x" * 2000)
    assert src.judge_one(lane, item) == ("philosophical", "r", None, PASSED)
    body = lane.bodies[0]
    assert body["temperature"] == 0 and body["top_p"] == 1.0 and body["presence_penalty"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == src.JUDGE_MAX_TOKENS
    assert "model" not in body   # the lane sets its operator route
    assert lane.timeouts == [300]


def test_judge_failed_call_is_an_error_tuple_and_recall_propagates(src) -> None:
    item = src.Item("outreach", "outreach:1", "q", "t")
    label, reason, err, model = src.judge_one(FakeLane(error=LaneCallError("http 502", 502)), item)
    assert (label, reason, model) == (None, None, None) and err.startswith("call: http 502")
    assert src.judge_one(FakeLane(None), item)[2].startswith("call: bad reply")
    with pytest.raises(LaneRecalled):
        src.judge_one(FakeLane(error=LaneRecalled("seat taken")), item)


def test_judge_unevaluated_model_is_an_error_not_a_label(src) -> None:
    item = src.Item("outreach", "outreach:1", "q", "t")
    assert src.judge_one(FakeLane(_reply(model="other-8b")), item) == (
        None, None, "model_not_evaluated: other-8b", "other-8b")


class _Grant:
    def __init__(self, model_file):
        self.model_file = model_file


class _Lease:
    def __init__(self, model_file):
        self.grant = _Grant(model_file)


def test_a_reply_that_echoes_the_route_reads_the_grants_model(src) -> None:
    # The lane sends model=operator_agent; a llama.cpp/strata seat may echo it. A route is not a model.
    item = src.Item("outreach", "outreach:1", "q", "t")
    lane = FakeLane(_reply(model="operator_agent"))
    lane.route, lane.lease = "operator_agent", _Lease("qwen3.8-flash-next-iq3_s.gguf")
    assert src.judge_one(lane, item) == ("operational", "r", None, "qwen3.8-flash-next-iq3_s.gguf")
    bare = FakeLane(_reply(model="operator_agent"))
    bare.route = "operator_agent"
    assert src.judge_one(bare, item)[2] == "model_not_evaluated: unknown"
    # A real served name in the reply wins over the grant record.
    named = FakeLane(_reply(model="other-8b"))
    named.route, named.lease = "operator_agent", _Lease("qwen3.8-flash-next-iq3_s.gguf")
    assert src.judge_one(named, item)[2] == "model_not_evaluated: other-8b"
