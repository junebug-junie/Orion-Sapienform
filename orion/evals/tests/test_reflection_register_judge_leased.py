"""The reflection-register judge (scripts/analysis/reflection_register_sources.judge_one) and its three
callers call only through an operator lane (orion/evals/leased_llm.py), never a seat URL (#3921: the door
replay sent 286+ unleased calls to the agent seat on 2026-11-25). No pool, model or network here."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from orion.evals.leased_llm import EXIT_RECALLED, LaneCallError, LaneRecalled

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


@pytest.fixture
def share(monkeypatch, src):
    return _load(SHARE, "eval_reflection_register_share", monkeypatch)


@pytest.fixture
def anchor(monkeypatch, src):
    return _load(ANCHOR, "judge_anchor_set", monkeypatch)


@pytest.fixture
def replay(monkeypatch, src):
    # The replay imports Hub's own ``scripts`` package; the repo root has one too. Drop any cached one.
    for name in [m for m in sys.modules if m == "scripts" or m.startswith("scripts.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(ROOT / "services/orion-hub"))
    return _load(REPLAY, "run_self_question_door_replay", monkeypatch)


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


def _fake_hold(lane: FakeLane, seen: list):
    @contextlib.contextmanager
    def hold(name, holder):
        seen.append((name, holder))
        yield lane
    return hold


@pytest.mark.parametrize("path", [SOURCES, SHARE, ANCHOR, REPLAY])
def test_no_direct_url_mode_and_no_worker_address(path: Path) -> None:
    src = path.read_text()
    assert "httpx" not in src and "urllib.request" not in src
    assert "/chat/completions" not in src
    assert not re.search(r"https?://[\w.\-<>]+:\d+", src)
    assert "judge-url" not in src and "judge_url" not in src


@pytest.mark.parametrize("path", [SHARE, ANCHOR, REPLAY])
def test_every_caller_takes_one_operator_hold(path: Path) -> None:
    src = path.read_text()
    assert "operator_lane_sync" in src and 'HOLDER = "operator:' in src


def test_judge_labels_through_the_lane_with_pinned_sampling(src) -> None:
    lane = FakeLane(_reply("philosophical"))
    item = src.Item("outreach", "outreach:1", "q", "x" * 2000)
    assert src.judge_one(lane, item) == ("philosophical", "r", None, PASSED)
    body = lane.bodies[0]
    assert body["temperature"] == 0 and body["top_p"] == 1.0 and body["presence_penalty"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == src.JUDGE_MAX_TOKENS
    assert "model" not in body   # the lane sets its operator route
    assert lane.timeouts == [src.JUDGE_TIMEOUT_SEC]
    assert src.DEFAULT_LANE == "agent"


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


@pytest.mark.parametrize("argv_mod", [("share", ["--judge-url", "http://h:1"]),
                                      ("share", ["--lane", "nope"]),
                                      ("anchor", ["score", "--judge-url", "http://h:1"]),
                                      ("replay", ["--live", "--judge-url", "http://h:1"])])
def test_old_url_flag_and_unknown_lanes_are_refused(argv_mod, request) -> None:
    mod = request.getfixturevalue(argv_mod[0])
    with pytest.raises(SystemExit):
        mod.main(argv_mod[1])


def test_share_recall_writes_partial_rows_and_exits_3(share, monkeypatch, tmp_path) -> None:
    items = [share.Item("outreach", f"outreach:{i}", "q", "t") for i in range(3)]
    monkeypatch.setattr(share, "read_items", lambda organ, **kw: items)
    seen: list = []
    monkeypatch.setattr(share, "operator_lane_sync", _fake_hold(FakeLane(_reply(), recall_after=2), seen))
    rc = share.main(["--organs", "outreach", "--out", str(tmp_path), "--end", "2026-11-25T00:00:00+00:00"])
    assert rc == EXIT_RECALLED and seen == [("agent", share.HOLDER)]
    rows = [json.loads(x) for x in (tmp_path / "items.partial.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(r["label"] == "operational" for r in rows)
    assert not (tmp_path / "items.jsonl").exists()


def test_share_stored_labels_only_take_no_hold(share, monkeypatch, tmp_path) -> None:
    items = [share.Item("self_inquiry_finding", "curiosity_run:1", "", "", stored_label="philosophical")]
    monkeypatch.setattr(share, "read_items", lambda organ, **kw: items)

    def no_hold(*a, **k):
        raise AssertionError("no hold for stored labels")

    monkeypatch.setattr(share, "operator_lane_sync", no_hold)
    assert share.main(["--organs", "self_inquiry_finding", "--out", str(tmp_path),
                       "--end", "2026-11-25T00:00:00+00:00"]) == 0
    assert "stored register labels only" in (tmp_path / "report.md").read_text()


def test_anchor_score_recall_exits_3_without_export(anchor, monkeypatch, tmp_path) -> None:
    tsv = tmp_path / "a.tsv"
    tsv.write_text("\t".join(anchor.HEADER) + "\n" + "".join(
        f"p\t{i}\toutreach\toutreach:{i}\tq\tt\n" for i in range(3)))
    seen: list = []
    monkeypatch.setattr(anchor, "operator_lane_sync", _fake_hold(FakeLane(_reply("philosophical"), recall_after=1), seen))
    export = tmp_path / "out.jsonl"
    assert anchor.main(["score", "--path", str(tsv), "--export", str(export)]) == EXIT_RECALLED
    assert seen == [("agent", anchor.HOLDER)] and not export.exists()


def test_replay_live_recall_writes_partial_questions_and_exits_3(replay, src, monkeypatch, tmp_path) -> None:
    qs = [{"id": f"lived.orion.{i}", "text": "why", "minted_by": "orion", "status": "open"} for i in range(3)]
    monkeypatch.setattr(src, "psql_json", lambda sql: qs)
    seen: list = []
    monkeypatch.setattr(replay, "operator_lane_sync", _fake_hold(FakeLane(_reply(), recall_after=2), seen))
    assert replay.main(["--live", "--out", str(tmp_path)]) == EXIT_RECALLED
    assert seen == [("agent", replay.HOLDER)]
    rows = [json.loads(x) for x in (tmp_path / "questions.partial.jsonl").read_text().splitlines()]
    assert [r["question_id"] for r in rows] == ["lived.orion.0", "lived.orion.1"]


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


def test_share_partial_report_counts_the_organ_cut_off(share, monkeypatch, tmp_path) -> None:
    items = [share.Item("outreach", f"outreach:{i}", "q", "t") for i in range(3)]
    monkeypatch.setattr(share, "read_items", lambda organ, **kw: items)
    monkeypatch.setattr(share, "operator_lane_sync", _fake_hold(FakeLane(_reply(), recall_after=2), []))
    share.main(["--organs", "outreach", "--out", str(tmp_path), "--end", "2026-11-25T00:00:00+00:00"])
    assert "| outreach (cut off by recall) | 2 | 0 | 2 |" in (tmp_path / "report.partial.md").read_text()


def test_replay_reads_before_it_holds_a_seat(replay, src, monkeypatch, tmp_path) -> None:
    def boom(sql):
        raise RuntimeError("psql down")

    seen: list = []
    monkeypatch.setattr(src, "psql_json", boom)
    monkeypatch.setattr(replay, "operator_lane_sync", _fake_hold(FakeLane(_reply()), seen))
    with pytest.raises(RuntimeError):
        replay.main(["--live", "--out", str(tmp_path)])
    assert seen == []
