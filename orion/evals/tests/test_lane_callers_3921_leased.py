"""The two committed scripts that sent unleased calls to the agent seat (#3921) now call only through an
operator lane (orion/evals/leased_llm.py). No pool, model or network here."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from orion.evals.leased_llm import LaneCallError, LaneRecalled

ROOT = Path(__file__).resolve().parents[3]
BRACE = ROOT / "services/orion-cortex-exec/evals/eval_json_task_turn_end_after_brace.py"
JUDGE = ROOT / "scripts/analysis/eval_curiosity_register_judge.py"


def _load(path: Path, name: str, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def brace(monkeypatch):
    # The eval imports cortex-exec's `app.executor` (heavy deps); only two strings are used from it.
    app = types.ModuleType("app")
    executor = types.ModuleType("app.executor")
    executor.JSON_TASK_TURN, executor.SYSTEM_TASK_TURN = "json turn", "generic turn"
    monkeypatch.setitem(sys.modules, "app", app)
    monkeypatch.setitem(sys.modules, "app.executor", executor)
    return _load(BRACE, "eval_json_task_turn_end_after_brace", monkeypatch)


@pytest.fixture
def judge(monkeypatch):
    return _load(JUDGE, "eval_curiosity_register_judge", monkeypatch)


class FakeLane:
    def __init__(self, reply=None, error=None):
        self.bodies: list[dict] = []
        self.reply, self.error = reply, error

    def chat(self, body, timeout_sec=None):
        self.bodies.append(body)
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.mark.parametrize("path", [BRACE, JUDGE])
def test_no_direct_url_mode_and_no_worker_address(path: Path) -> None:
    src = path.read_text()
    assert "urllib.request" not in src and "httpx" not in src
    assert not re.search(r"https?://[\w.\-]+:\d+", src)
    assert not re.search(r"--url\b|base_url", src)
    assert "operator_lane_sync" in src


def test_brace_eval_sends_its_logprob_probe_through_the_lane(brace) -> None:
    reply = {"choices": [{"logprobs": {"content": [
        {"token": "{"}, {"token": "x", "top_logprobs": [{"token": "<|im_end|>", "logprob": -0.6931}]}]}}]}
    lane = FakeLane(reply)
    p = brace.p_end_after_brace(lane, {"messages": [{"role": "user", "content": "u"}]})
    assert p == pytest.approx(0.5, abs=1e-3)
    sent = lane.bodies[0]
    assert sent["logprobs"] is True and sent["top_logprobs"] == 20 and sent["max_tokens"] == 2
    assert sent["logit_bias"] == {brace.OPEN_BRACE_QUOTE_ID: -100}
    assert brace.HOLDER.startswith("operator:")


def test_brace_eval_reads_an_ended_reply(brace) -> None:
    lane = FakeLane({"choices": [{"logprobs": {"content": [{"token": "{"}]}}]})
    assert brace.p_end_after_brace(lane, {"messages": []}) == "ended"


def test_brace_eval_refuses_the_old_url_flag(brace) -> None:
    with pytest.raises(SystemExit):
        brace.main(["envelope.json", "--url", "http://127.0.0.1:8015"])


def test_judge_labels_a_row_through_the_lane(judge) -> None:
    lane = FakeLane({"model": "m", "choices": [{"message": {"content": json.dumps(
        {"register": "operational", "reason": "r"})}}]})
    row = judge.judge_one(lane, {"run_id": "r1", "question": "q", "answer": "a"})
    assert row["error"] is None and row["model"] == "m"
    body = lane.bodies[0]
    assert body["temperature"] == 0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "model" not in body   # the lane sets its operator route
    assert judge.HOLDER.startswith("operator:")


def test_judge_failed_call_is_a_row_error_and_recall_stops(judge) -> None:
    it = {"run_id": "r1", "question": "q", "answer": "a"}
    row = judge.judge_one(FakeLane(error=LaneCallError("http 502", 502)), it)
    assert row["label"] is None and row["error"].startswith("call: http 502")
    with pytest.raises(LaneRecalled):
        judge.judge_one(FakeLane(error=LaneRecalled("seat taken")), it)
    assert judge.judge_one(FakeLane(), {"missing": True})["error"] == "no_input"


def test_judge_lane_names_are_operator_lanes(judge) -> None:
    with pytest.raises(SystemExit):
        judge.main(["--lane", "metacog=http://127.0.0.1:8012", "--out", "/nonexistent"])
