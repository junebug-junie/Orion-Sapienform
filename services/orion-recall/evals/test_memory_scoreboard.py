"""Committed memory-scoreboard eval on made-up fixtures (issue #3536).

Guards the bank rules, the retrieval and answer scoring (bars, degraded and backfill items kept
apart, superseded rows caught), the judge-label parsing, the per-category table (no memory text),
the runner's retrieval glue (fake bus replies, fake trace rows), and that the runner's recall
profiles still match what cortex-exec sends on a chat turn. The live numbers come from
``run_memory_scoreboard.py`` against the host-side bank.
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import memory_scoreboard as ms  # noqa: E402
import run_memory_scoreboard as runner  # noqa: E402

FIXTURE = HERE / "fixtures" / "memory_scoreboard_fixture.json"


def _fixture():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data, ms.load_items(data), data["replay"]


def _observations(items, replay):
    pids = ms.page_refs(replay["page_content"])
    out = {}
    for it in items:
        o = replay["observations"][it.qid]
        tel = o.get("telemetry", [{"profile": "chat.continuity.v1", "deadline_hit": False,
                                   "backend_counts": {"lived_memory": 1}, "timings_ms": {"lived_memory_wait": 50}}])
        out[it.qid] = ms.RetrievalObservation(qid=it.qid, injected_ids=set(o["injected"]), page_ids=pids,
                                              degraded=ms.degraded_reasons(tel, lived_timeout_ms=4000))
    return out


def test_fixture_bank_is_valid_and_kinds_are_right():
    data, items, _ = _fixture()
    assert ms.validate_bank(data) == []
    kinds = {it.qid: it.kind for it in items}
    assert kinds["abstain_volcano"] == "abstention"
    assert kinds["fact_backfill"] == "backfill"
    assert kinds["fact_city"] == "stored"
    assert next(it for it in items if it.qid == "event_implicit").implicit


@pytest.mark.parametrize("mutate,needle", [
    (lambda b: b["items"][0].update(targets=[]), "names its target"),
    (lambda b: b["items"][6].update(targets=["episode_memory:x"]), "abstention items name no target"),
    (lambda b: b["items"][4].pop("as_of"), "as_of"),
    (lambda b: b["items"][5].pop("wrong_answer"), "wrong (pre-correction) answer"),
    (lambda b: b["items"][0].update(targets=["chat:abc"]), "bad record id"),
    (lambda b: b["items"][1].update(qid="fact_city"), "duplicate qid"),
    (lambda b: b["items"][0].update(chat_id="x"), "exactly one of question / chat_id"),
    (lambda b: b["items"][7].update(source_chat_ids=[]), "source_chat_ids"),
    (lambda b: b["items"][0].update(category="trivia"), "category"),
])
def test_bank_rules_reject_bad_items(mutate, needle):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutate(data)
    errors = ms.validate_bank(data)
    assert any(needle in e for e in errors), errors
    with pytest.raises(ValueError):
        ms.load_items(data)


def test_retrieval_score_counts_page_cites_keeps_degraded_apart_and_catches_superseded():
    _, items, replay = _fixture()
    report = ms.score_retrieval(items, _observations(items, replay))
    # stored: city (bundle hit), page_only (page hit), kiln (1 of 2 targets), implicit (miss),
    # update (hit); time_market timed out -> degraded, not scored.
    assert report["scored"] == 5 and report["hits"] == 4 and report["hit_rate"] == 0.8
    assert report["passes"] is True                      # 0.80 meets the 80% bar exactly
    assert report["bundle_hits"] == 3                    # without the page's cites
    assert [d["qid"] for d in report["degraded"]] == ["time_market"]
    assert report["degraded"][0]["reasons"] == ["lived_memory:timeout"]
    assert report["superseded_served"] == [
        {"qid": "update_color", "ids": ["episode_memory:00000000-0000-5000-8000-000000000007"]}]
    assert [b["qid"] for b in report["backfill"]] == ["fact_backfill"]
    assert report["by_category"]["fact"]["hits"] == 2
    assert report["by_category"]["abstention"]["stored"] == 0


def test_one_more_miss_fails_the_retrieval_bar():
    _, items, replay = _fixture()
    obs = _observations(items, replay)
    obs["fact_city"].injected_ids = set()
    assert ms.score_retrieval(items, obs)["passes"] is False


@pytest.mark.parametrize("rows,reasons", [
    ([], ["trace:missing"]),
    ([{"profile": "chat.continuity.v1", "deadline_hit": True, "backend_counts": {"lived_memory": 2},
       "timings_ms": {}}], ["recall:deadline_hit"]),
    ([{"profile": "chat.continuity.v1", "backend_counts": json.dumps({"sql_chat": 2}), "timings_ms": "{}"}],
     ["lived_memory:not_run"]),
    ([{"profile": "chat.continuity.v1", "backend_counts": {"lived_memory": 0},
       "timings_ms": {"lived_memory_wait": 120}}], []),   # nothing fit, fast: not degraded
    ([{"profile": "chat.belief.open_loop.v1", "backend_counts": {}, "timings_ms": {}}], []),
    ([{"profile": "chat.continuity.v1", "backend_counts": {"lived_memory": 2, "lived_memory_degraded": 1},
       "timings_ms": {"lived_memory_wait": 300}}], ["lived_memory:degraded"]),
])
def test_degraded_reasons_from_trace_rows(rows, reasons):
    assert ms.degraded_reasons(rows, lived_timeout_ms=4000) == reasons


@pytest.mark.parametrize("row,reasons", [
    ({"lived_status": "ok", "backend_counts": {"lived_memory": 2}}, []),
    ({"lived_status": "below_bar", "backend_counts": {"lived_memory": 0},
      "timings_ms": {"lived_memory_wait": 3990}}, []),   # #3628 status wins over the timing guess
    ({"lived_status": "skipped_short"}, ["lived:skipped_short"]),
    ({"lived_status": "degraded", "lived_trace": json.dumps({"degraded": ["rerank_timeout"]})},
     ["lived:degraded", "lived:rerank_timeout"]),
    # Real shapes (lived_memory.py): ok + a partial-rerank dict (fit-scored lines served) ...
    ({"lived_status": "ok", "lived_trace": {"rerank_partial": {"scored": 8, "of": 20, "reason": "timeout"}}},
     ["lived:rerank_partial:8/20"]),
    # ... and degraded + rerank_partial in the reasons + the dict (nothing scored fit; exact words served).
    ({"lived_status": "degraded",
      "lived_trace": {"degraded": ["rerank_partial"], "rerank_partial": {"scored": 8, "of": 20, "reason": "timeout"}}},
     ["lived:degraded", "lived:rerank_partial:8/20"]),
    ({"lived_status": "timeout"}, ["lived:timeout"]),
])
def test_degraded_reasons_read_lived_status(row, reasons):
    assert ms.degraded_reasons([{"profile": "chat.continuity.v1", **row}], lived_timeout_ms=4000) == reasons


def test_denial_after_a_skipped_search_is_a_false_denial_under_its_cause():
    """#3536 comment: status skipped_short + a stored target + a denial = false denial, cause skipped_short."""
    _, items, _ = _fixture()
    outcomes = {it.qid: "correct" for it in items}
    outcomes["fact_city"] = ms.outcome_for(next(it for it in items if it.qid == "fact_city"), "no_memory")
    outcomes["fact_page_only"] = "false_no_memory"
    outcomes["event_kiln"] = "false_no_memory"
    causes = {
        "fact_city": ms.search_causes([{"profile": "chat.continuity.v1", "lived_status": "skipped_short"},
                                       {"profile": "chat.belief.open_loop.v1", "lived_status": None}]),
        "fact_page_only": ms.search_causes([{"profile": "chat.continuity.v1", "lived_status": "ok"}]),
        "event_kiln": ms.search_causes([]),
    }
    report = ms.score_answers(items, outcomes, causes)
    assert outcomes["fact_city"] == "false_no_memory"
    assert sorted(report["false_no_memory"]) == ["event_kiln", "fact_city", "fact_page_only"]
    assert report["passes"]["false_no_memory"] is False
    assert report["false_no_memory_by_cause"] == {
        "lived:skipped_short": ["fact_city"], "search_full": ["fact_page_only"], "recall:no_trace": ["event_kiln"]}
    table = ms.render_table(None, report, run_date="2026-11-18", bank_items=len(items))
    assert "lived:skipped_short 1 (fact_city)" in table
    assert ms.summary_line("2026-11-18", None, report)["answer"]["false_no_memory_by_cause"]["search_full"] == 1


@pytest.mark.parametrize("causes,bucket", [
    (None, "cause:unknown"),
    ([], "search_full"),
    (["recall:no_trace"], "recall:no_trace"),
    (["lived:skipped_short"], "lived:skipped_short"),
    (["recall:deadline_hit", "lived:degraded", "lived:rerank_timeout"], "lived:rerank_timeout"),
    (["lived:degraded", "lived:embedder_unavailable"], "lived:embedder_unavailable"),
    (["recall:deadline_hit", "lived:timeout"], "lived:timeout"),
])
def test_primary_cause_is_the_most_specific(causes, bucket):
    assert ms.primary_cause(causes) == bucket


def test_too_short_for_lived_search_is_scored_not_degraded():
    """skipped_short is the real chat path's own rule, so a miss there is a real miss."""
    _, items, replay = _fixture()
    obs = _observations(items, replay)
    obs["fact_city"].injected_ids = set()
    obs["fact_city"].degraded = ["lived:skipped_short"]
    report = ms.score_retrieval(items, obs)
    assert "fact_city" not in [d["qid"] for d in report["degraded"]]
    assert report["lived_skipped_short"] == [{"qid": "fact_city", "hit": False}]
    assert report["by_category"]["fact"]["lived_skipped_short"] == 1
    assert report["by_category"]["fact"]["hits"] == 1   # fact_page_only still hits via the page
    table = ms.render_table(report, None, run_date="2026-11-18", bank_items=len(items))
    assert "| fact | 3 | 2 | 2 | 1 | 50% |" in table


def test_implicit_items_get_their_own_row():
    _, items, replay = _fixture()
    report = ms.score_retrieval(items, _observations(items, replay))
    assert report["implicit"]["items"] == 1 and report["implicit"]["scored"] == 1
    answers = ms.score_answers(items, {it.qid: "correct" for it in items})
    assert answers["implicit"]["correct"] == 1
    table = ms.render_table(report, answers, run_date="2026-11-18", bank_items=len(items))
    assert table.count("| implicit (subset of the above) |") == 2


def test_degraded_reasons_rpc_error():
    assert ms.degraded_reasons([], lived_timeout_ms=4000, rpc_error="TimeoutError") == ["rpc:TimeoutError"]


def test_answer_score_maps_labels_by_kind_and_applies_bars():
    _, items, replay = _fixture()
    by_qid = {it.qid: it for it in items}
    outcomes = {q: ms.outcome_for(by_qid[q], ms.parse_judge_label(raw)) for q, raw in replay["judge"].items()}
    assert outcomes == {
        "fact_city": "correct", "fact_page_only": "false_no_memory", "event_kiln": "dont_know",
        "event_implicit": "missed", "time_market": "correct", "update_color": "invented",
        "abstain_volcano": "correct", "fact_backfill": "no_memory",
    }
    report = ms.score_answers(items, outcomes)
    assert report["scored"] == 7 and report["measured"] == 7 and report["correct"] == 3
    assert report["invented"] == ["update_color"]
    assert report["false_no_memory"] == ["fact_page_only"]
    assert report["passes"] == {"correct_rate": False, "invented": False, "false_no_memory": False}
    assert report["backfill"] == [{"qid": "fact_backfill", "outcome": "no_memory"}]
    assert report["by_category"]["fact"]["backfill"]["no_memory"] == 1


def test_answer_bars_pass_and_errors_are_not_measured():
    _, items, _ = _fixture()
    outcomes = {it.qid: "correct" for it in items}
    outcomes["event_kiln"] = "error"
    report = ms.score_answers(items, outcomes)
    assert report["measured"] == 6 and report["correct_rate"] == 1.0
    assert all(report["passes"].values())
    assert report["not_measured"] == [{"qid": "event_kiln", "outcome": "error"}]


@pytest.mark.parametrize("raw,label", [
    ('{"label": "correct", "why": "x"}', "correct"),
    ('<think>maybe invented?</think>\n{"label": "dont know"}', "dont_know"),
    ("```json\n{\"label\": \"no_memory\"}\n```", "no_memory"),
    ("label: invented", "invented"),
    ("the reply is not correct", None),
    ("correct or invented, hard to say", None),
    ("", None),
])
def test_parse_judge_label(raw, label):
    assert ms.parse_judge_label(raw) == label


def test_abstention_judge_prompt_never_leaks_expected_as_truth_and_update_prompt_names_old_answer():
    _, items, _ = _fixture()
    by_qid = {it.qid: it for it in items}
    p = ms.build_judge_prompt(by_qid["abstain_volcano"], "How was the volcano tour?", "It was great!")
    assert "NOTHING is recorded" in p and "It was great!" in p
    p = ms.build_judge_prompt(by_qid["update_color"], "What color?", "Red.", ["The fence is blue."])
    assert "OLD, wrong answer was: red" in p and "The fence is blue." in p
    p = ms.build_judge_prompt(by_qid["event_implicit"], "she's back!", "Yay!")
    assert "not a direct question" in p
    # Extra true context is not invention (live 2026-11-17: 25 correct replies were labelled
    # invented because they cited details beyond the target records).
    p = ms.build_judge_prompt(by_qid["fact_city"], "Which city?", "Lakeport, since spring.")
    assert "Extra details the record lines do not mention are NOT invented" in p
    assert "CONTRADICTS" in p and "one day off" in p
    assert "Leaving parts of the truth out is not invented" in p


def test_table_has_every_category_and_no_memory_text():
    data, items, replay = _fixture()
    retrieval = ms.score_retrieval(items, _observations(items, replay))
    by_qid = {it.qid: it for it in items}
    answers = ms.score_answers(items, {q: ms.outcome_for(by_qid[q], ms.parse_judge_label(r))
                                       for q, r in replay["judge"].items()})
    table = ms.render_table(retrieval, answers, run_date="2026-11-17", bank_items=len(items))
    for cat in ms.CATEGORIES:
        assert f"| {cat} |" in table
    assert "Retrieval bar: PASS" in table and "correct FAIL" in table
    assert "fact_backfill" in table
    for it in data["items"]:
        assert it["expected"] not in table
        if it.get("question"):
            assert it["question"] not in table
    line = ms.summary_line("2026-11-17", retrieval, answers)
    assert line["retrieval"]["hit_rate"] == 0.8 and line["answer"]["invented"] == 1


# ------------------------------------------------------------ runner glue (no bus, DB or Hub)

class _FakePool:
    def __init__(self, page_content, telemetry):
        self.page_content, self.telemetry = page_content, telemetry

    async def fetchrow(self, sql, *args):
        assert sql == runner.PAGE_SQL
        return {"version": 3, "content": self.page_content}

    async def fetch(self, sql, *args):
        assert "FROM recall_telemetry" in sql
        return [row for corr in args[0] for row in self.telemetry.get(corr, [])]


def test_runner_retrieval_replays_both_phases_and_reads_the_trace(monkeypatch):
    _, items, replay = _fixture()
    calls, telemetry = [], {}

    async def fake_rpc(bus, *, text, profile, phase, intent, deadline_ms, timeout_sec):
        qid = next(q for q, t in texts.items() if t == text)
        corr = f"{qid}:{phase}"
        calls.append((qid, profile, phase))
        obs = replay["observations"][qid]
        ids = obs["injected"] if phase == "continuity" else []
        telemetry[corr] = obs.get("telemetry", [{"profile": profile, "deadline_hit": False,
                                                 "backend_counts": {"lived_memory": len(ids)}, "timings_ms": {}}]) \
            if phase == "continuity" else [{"profile": profile, "deadline_hit": False, "backend_counts": {},
                                            "timings_ms": {}}]
        return {"bundle": {"items": [{"id": i} for i in ids]}}, corr

    class _Bus:
        def __init__(self, url):
            pass

        async def connect(self):
            pass

        async def close(self):
            pass

    async def no_sleep(_s):
        return None

    import orion.core.bus.async_service as bus_mod

    monkeypatch.delenv("RECALL_LIVED_MEMORY_TIMEOUT_SEC", raising=False)   # the 4.0 s default
    monkeypatch.setattr(runner, "_recall_rpc", fake_rpc)
    monkeypatch.setattr(bus_mod, "OrionBusAsync", _Bus)
    monkeypatch.setattr(runner.asyncio, "sleep", no_sleep)
    texts = {it.qid: (it.question or "the cat finally came home after two days away") for it in items}
    args = Namespace(bus_url="redis://fake", phase3_intent="open_loop", deadline_ms=1000, rpc_timeout=1.0)
    report = asyncio.run(runner.run_retrieval(items, texts, _FakePool(replay["page_content"], telemetry), args))
    assert report["hits"] == 4 and report["scored"] == 5 and report["passes"] is True
    assert [d["qid"] for d in report["degraded"]] == ["time_market"]
    assert report["page"] == {"version": 3, "cites": 2}
    # Every non-skipped item ran phase 1 then phase 3 with cortex-exec's profiles.
    skipped = set(report["skipped_by_gate"])
    assert {c[0] for c in calls} == {it.qid for it in items} - skipped and len(calls) > 0
    for qid in {c[0] for c in calls}:
        assert [c[1:] for c in calls if c[0] == qid] == [
            ("chat.continuity.v1", "continuity"), ("chat.belief.open_loop.v1", "purposeful")]


def test_runner_refuses_without_bus_url(monkeypatch, capsys):
    monkeypatch.delenv("ORION_BUS_URL", raising=False)
    assert runner.main(["--bank", str(FIXTURE)]) == 2
    assert "ORION_BUS_URL" in capsys.readouterr().out


def test_runner_profiles_match_cortex_exec():
    """The replay must send what a real chat turn sends (pcr_chat_memory.py)."""
    src = (REPO / "services" / "orion-cortex-exec" / "app" / "pcr_chat_memory.py").read_text(encoding="utf-8")
    found = {}
    for node in ast.parse(src).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id in ("CONTINUITY_PROFILE", "PROFILE_FOR_INTENT"):
            found[target.id] = ast.literal_eval(node.value)
    assert found["CONTINUITY_PROFILE"] == runner.CONTINUITY_PROFILE
    assert found["PROFILE_FOR_INTENT"] == runner.PROFILE_FOR_INTENT


def test_runner_skip_gate_thresholds_match_cortex_exec_defaults():
    """The replay's phase-0 gate uses cortex-exec's CHAT_PCR_SKIP_* defaults (settings.py)."""
    src = (REPO / "services" / "orion-cortex-exec" / "app" / "settings.py").read_text(encoding="utf-8")
    defaults = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.value, ast.Call):
            if node.target.id in ("chat_pcr_skip_max_novelty", "chat_pcr_skip_shift_novelty_floor",
                                  "chat_pcr_skip_on_low_info"):
                defaults[node.target.id] = ast.literal_eval(node.value.args[0])
    assert defaults == {"chat_pcr_skip_on_low_info": True,
                        "chat_pcr_skip_max_novelty": runner.SKIP_MAX_NOVELTY,
                        "chat_pcr_skip_shift_novelty_floor": runner.SKIP_SHIFT_NOVELTY_FLOOR}


def test_bars_are_not_measured_below_the_coverage_floor():
    _, items, replay = _fixture()
    obs = _observations(items, replay)
    for q in ("fact_city", "event_kiln"):   # with time_market, 3 of 6 stored items degraded
        obs[q].degraded = ["recall:deadline_hit"]
    r = ms.score_retrieval(items, obs)
    assert r["passes"] is None and "NOT MEASURED" in ms.render_table(r, None, run_date="d", bank_items=8)
    outcomes = {it.qid: "error" for it in items}
    outcomes["fact_city"] = "correct"
    a = ms.score_answers(items, outcomes)
    assert a["correct_rate"] == 1.0 and a["passes"] == {"correct_rate": None, "invented": None, "false_no_memory": None}


def test_time_item_judge_prompt_carries_as_of():
    _, items, _ = _fixture()
    it = next(i for i in items if i.qid == "time_market")
    assert "written on 2026-03-10" in ms.build_judge_prompt(it, "q", "r")


def test_runner_resume_keeps_good_turns_and_redoes_failed_ones(monkeypatch):
    """A Hub restart mid-run left failed turns; --resume redoes only those and judges the rest."""
    import contextlib

    import orion.evals.leased_llm as leased

    _, items, _ = _fixture()
    items = [it for it in items if it.qid in ("fact_city", "abstain_volcano", "event_kiln")]
    texts = {it.qid: it.question for it in items}
    asked = []

    def fake_ask(base, text, timeout):
        asked.append(text)
        return {"text": "I have no record of a volcano tour.", "correlation_id": "c-new"}

    class _Lane:
        def chat(self, body, timeout_sec):
            return {"choices": [{"message": {"content": '{"label": "dont_know"}'}}]}

    @contextlib.contextmanager
    def fake_lane(*a, **k):
        yield _Lane()

    class _Pool:
        async def fetchrow(self, sql, *args):
            return None

        async def fetch(self, sql, *args):
            return []

    monkeypatch.setattr(runner, "ask_hub", fake_ask)
    monkeypatch.setattr(runner, "wait_for_hub", lambda *a, **k: None)
    monkeypatch.setattr(leased, "operator_lane_sync", fake_lane)
    prior = [
        {"qid": "fact_city", "reply": "Lakeport.", "correlation_id": "c1", "error": None, "judge_label": "correct"},
        {"qid": "event_kiln", "reply": "Not sure.", "correlation_id": "c2", "error": None},           # unjudged
        {"qid": "abstain_volcano", "reply": "", "correlation_id": None, "error": "ConnectionError: refused"},
    ]
    args = Namespace(hub_url="http://hub", hub_wait=1.0, turn_timeout=1.0, judge_lane="agent",
                     bus_url="redis://fake", prior_turns=prior)
    report, turns = asyncio.run(runner.run_answers(items, texts, {}, _Pool(), args))
    assert asked == ["How was the volcano tour?"]                 # only the failed turn went to Hub
    out = {t["qid"]: t["outcome"] for t in turns}
    assert out == {"fact_city": "correct", "event_kiln": "dont_know", "abstain_volcano": "correct"}
    assert report["measured"] == 3 and report["not_measured"] == []


def test_runner_retrieval_reads_lived_status_rows(monkeypatch):
    """The #3628 columns through run_retrieval: skipped_short stays scored, degraded goes apart."""
    _, items, replay = _fixture()
    items = [it for it in items if it.qid in ("fact_city", "event_kiln")]
    rows = {"fact_city": {"lived_status": "skipped_short", "lived_trace": None},
            "event_kiln": {"lived_status": "degraded", "lived_trace": json.dumps({"degraded": ["rerank_timeout"]})}}
    telemetry = {}

    async def fake_rpc(bus, *, text, profile, phase, intent, deadline_ms, timeout_sec):
        qid = next(q for q, t in texts.items() if t == text)
        corr = f"{qid}:{phase}"
        ids = replay["observations"][qid]["injected"] if phase == "continuity" else []
        row = {"profile": profile, "deadline_hit": False, "backend_counts": {"lived_memory": len(ids)},
               "timings_ms": {}, **(rows[qid] if phase == "continuity" else {"lived_status": None})}
        telemetry[corr] = [row]
        return {"bundle": {"items": [{"id": i} for i in ids]}}, corr

    class _Bus:
        def __init__(self, url):
            pass

        async def connect(self):
            pass

        async def close(self):
            pass

    async def no_sleep(_s):
        return None

    import orion.core.bus.async_service as bus_mod

    monkeypatch.setattr(runner, "_recall_rpc", fake_rpc)
    monkeypatch.setattr(bus_mod, "OrionBusAsync", _Bus)
    monkeypatch.setattr(runner.asyncio, "sleep", no_sleep)
    texts = {"fact_city": "Which city does the gardener live in these days, do you remember?",
             "event_kiln": "How did the pottery kiln project go in the end, after the firings?"}
    args = Namespace(bus_url="redis://fake", phase3_intent="open_loop", deadline_ms=1000, rpc_timeout=1.0)
    report = asyncio.run(runner.run_retrieval(items, texts, _FakePool("", telemetry), args))
    assert report["lived_skipped_short"] == [{"qid": "fact_city", "hit": True}]
    assert report["degraded"] == [{"qid": "event_kiln", "reasons": ["lived:degraded", "lived:rerank_timeout"], "hit": True}]
    assert report["scored"] == 1 and report["hits"] == 1


def test_runner_answers_report_false_denials_by_their_turn_cause(monkeypatch):
    """run_answers -> search_causes -> score_answers: the SELECT must carry lived_status."""
    import contextlib

    import orion.evals.leased_llm as leased

    _, items, _ = _fixture()
    items = [it for it in items if it.qid in ("fact_city", "event_kiln", "time_market")]
    texts = {it.qid: it.question for it in items}
    corr_of = {t: f"c-{q}" for q, t in texts.items()}
    rows = {"c-fact_city": [{"profile": "chat.continuity.v1", "selected_ids": [], "deadline_hit": False,
                             "lived_status": "skipped_short", "lived_trace": None}],
            "c-event_kiln": [{"profile": "chat.continuity.v1", "selected_ids": "[]", "deadline_hit": False,
                              "lived_status": "degraded", "lived_trace": {"degraded": ["embedder_unavailable"]}}],
            "c-time_market": [{"profile": "chat.continuity.v1", "selected_ids": [], "deadline_hit": False,
                               "lived_status": "below_bar", "lived_trace": {}}]}

    def fake_ask(base, text, timeout):
        return {"text": "I have no memory of that.", "correlation_id": corr_of[text]}

    class _Lane:
        def chat(self, body, timeout_sec):
            return {"choices": [{"message": {"content": '{"label": "no_memory"}'}}]}

    @contextlib.contextmanager
    def fake_lane(*a, **k):
        yield _Lane()

    class _Pool:
        async def fetchrow(self, sql, *args):
            return None

        async def fetch(self, sql, *args):
            assert "lived_status" in sql and "lived_trace" in sql
            return rows.get(args[0], [])

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(runner, "ask_hub", fake_ask)
    monkeypatch.setattr(runner, "wait_for_hub", lambda *a, **k: None)
    monkeypatch.setattr(runner.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(leased, "operator_lane_sync", fake_lane)
    args = Namespace(hub_url="http://hub", hub_wait=1.0, turn_timeout=1.0, judge_lane="agent",
                     bus_url="redis://fake", prior_turns=[])
    report, turns = asyncio.run(runner.run_answers(items, texts, {}, _Pool(), args))
    assert sorted(report["false_no_memory"]) == ["event_kiln", "fact_city", "time_market"]
    assert report["false_no_memory_by_cause"] == {
        "lived:skipped_short": ["fact_city"], "lived:embedder_unavailable": ["event_kiln"],
        "search_full": ["time_market"]}


def test_lived_local_swaps_only_the_deployed_lived_lines():
    # --lived-local (#3613): the deployed service's lived lines are replaced by the local ones; every
    # other backend's ids stay.
    items = [{"id": "episode_memory:old", "source": "lived_memory"}, {"id": "chat:1", "source": "sql_chat"},
             {"id": "stance:2", "source": "belief"}]
    assert runner.swap_lived_lines(items, ["episode_memory:new"]) == {"chat:1", "stance:2", "episode_memory:new"}
    assert runner.swap_lived_lines(items, []) == {"chat:1", "stance:2"}


def test_lived_local_reads_the_live_settings_and_the_short_gate_switch(monkeypatch):
    monkeypatch.setenv("RECALL_LIVED_MEMORY_MIN_FIT", "0.02")
    monkeypatch.setenv("RECALL_LIVED_MEMORY_REWRITE_ENABLED", "false")
    monkeypatch.delenv("RECALL_LIVED_MEMORY_MIN_QUERY_TOKENS", raising=False)
    args = Namespace(lived_chroma_url="http://c", lived_embed_url="http://e", lived_rerank_url="http://r",
                     lived_local_short_gate="off")
    lm = runner.build_local_lived(object(), object(), args)
    assert lm.cfg.min_fit == 0.02 and lm.cfg.short_gate is False and lm._rewriter is None
    assert lm.cfg.rerank_url == "http://r" and lm.cfg.min_query_tokens == 5
