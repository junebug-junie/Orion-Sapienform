"""Full-window compactor passes: every PR / turn covered, journal body untrimmed, map-reduce over
the char budget. The digest calls run in a ``compactor.digest`` durable run, simulated in-line here
(tests/compactor_durable_sim.py); attempts / pool waits / resume are tested in orion-durable-runs."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from app import workflow_runtime as wr
from app.workflow_runtime import execute_chat_workflow
from orion.cognition.compactor import map_reduce
from orion.cognition.compactor.constants import DIGEST_MAX_TOKENS
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.contracts import CortexClientRequest


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, channel: str, envelope: Any) -> None:
        self.published.append((channel, envelope))

    def journal_bodies(self) -> list[str]:
        out = []
        for channel, env in self.published:
            if channel != "orion:journal:write":
                continue
            payload = env.payload if isinstance(env.payload, dict) else env.payload.model_dump(mode="json")
            out.append(payload["body"])
        return out


class _Result:
    def __init__(self, *, ok: bool = True, final_text: str = "", metadata: dict | None = None, error: str | None = None):
        self.ok = ok
        self.error = error
        self.output = {"result": {"status": "success", "final_text": final_text, "metadata": metadata or {}}}


def _req(workflow_id: str, **workflow_extra) -> CortexClientRequest:
    return CortexClientRequest.model_validate(
        {
            "mode": "brain",
            "route_intent": "none",
            "context": {
                "messages": [],
                "raw_user_text": "workflow please",
                "user_message": "workflow please",
                "session_id": "sid-1",
                "user_id": "user-1",
                "trace_id": "trace-1",
                "metadata": {
                    "workflow_request": {
                        "workflow_id": workflow_id,
                        "matched_alias": workflow_id,
                        "normalized_prompt": workflow_id,
                        "confidence": 1.0,
                        "resolver": "alias_registry",
                        **workflow_extra,
                    }
                },
            },
            "options": {},
            "packs": [],
            "recall": {"enabled": False, "required": False},
        }
    )


def _run(req, fake, bus):
    return asyncio.run(
        execute_chat_workflow(
            bus=bus,
            source=ServiceRef(name="cortex-orch"),
            req=req,
            correlation_id="00000000-0000-0000-0000-00000000c0de",
            causality_chain=[],
            trace={},
            call_verb_runtime=fake,
        )
    )


@pytest.fixture(autouse=True)
def _no_card_persist(monkeypatch):
    async def _persist(**kwargs):
        return "00000000-0000-0000-0000-0000000000aa"

    monkeypatch.setattr(wr, "persist_github_compactor_memory_card", _persist)
    monkeypatch.setattr(wr, "persist_chat_history_compactor_memory_card", _persist)


# ---------------------------------------------------------------- GitHub

def _forty_prs(merged_at: str = "2026-09-28T18:00:00Z") -> list[dict]:
    # ~9k-char bodies, like the real PR reports (median ~9-10k live 2026-09-26..29):
    # 40 of them (~360k chars) cannot fit one 100k-char call.
    return [
        {
            "number": 1000 + i,
            "title": f"PR {i}",
            "body": (f"Section for PR {1000 + i}. " + "detail " * 1300).strip(),
            "merged_at": merged_at,
            "url": f"https://github.com/acme/widgets/pull/{1000 + i}",
            "inferred_services": ["orion-hub"],
        }
        for i in range(40)
    ]


def _github_fetch_result(items: list[dict], **extra) -> _Result:
    payload = {"available": True, "repo": "acme/widgets", "lookback_days": 1, "merged_pr_count": len(items), "items": items, **extra}
    return _Result(final_text=json.dumps(payload))


def test_github_forty_prs_all_covered_via_map_reduce_and_body_untrimmed() -> None:
    bus = _Bus()
    digest_inputs: list[dict] = []
    digest_options: list[dict] = []
    long_body = "Merged narrative. " + ("every PR accounted for; " * 2000)  # ~48k chars, old cap 8000

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            return _github_fetch_result(_forty_prs())
        assert req.verb == "github_compactor_digest_v1"
        gi = req.context.metadata["github_compactor_input"]
        digest_inputs.append(gi)
        digest_options.append(dict(req.options))
        if "partial_digests" in gi:
            refs = [ref for part in gi["partial_digests"] for ref in part["pr_refs"]]
            digest = {"card_summary": "Day of 40 merges.", "journal_title": "Repo day", "journal_body": long_body, "pr_refs": refs}
        else:
            refs = [f"#{item['number']}" for item in gi["items"]]
            digest = {"card_summary": "part", "journal_title": "part", "journal_body": " ".join(refs), "pr_refs": refs}
        return _Result(final_text=json.dumps(digest))

    result = _run(_req("github_compactor_pass"), fake, bus)
    wf = result.metadata["workflow"]
    chunk_inputs = [gi for gi in digest_inputs if "items" in gi]
    covered = sorted(item["number"] for gi in chunk_inputs for item in gi["items"])
    assert covered == [1000 + i for i in range(40)]
    assert len(chunk_inputs) > 1
    assert all(gi["chunk_count"] == len(chunk_inputs) for gi in chunk_inputs)
    # No PR body was cut below the safety cap.
    assert all("truncated" not in item for gi in chunk_inputs for item in gi["items"])
    assert any("partial_digests" in gi for gi in digest_inputs)
    assert wf["total_count"] == wf["covered_count"] == wf["merged_pr_count"] == 40
    assert wf["input_truncated"] is False
    assert wf["digest_chunk_count"] == len(chunk_inputs)
    assert wf["digest_merge_mode"] == "llm_merge"
    # Journal body stored in full (the old 8000-char trim is gone).
    assert bus.journal_bodies() == [long_body]
    assert wf["journal_body_chars"] == len(long_body)
    # Explicit durable-admission route + completion budget on every call, and every call carries
    # the durable run's GPU pool hold (the gateway attaches it instead of queueing behind it).
    assert {o["llm_route"] for o in digest_options} == {"agent"}
    assert {o["max_tokens"] for o in digest_options} == {DIGEST_MAX_TOKENS}
    assert all(o["gpu_lease"]["lease_id"] == "hold-sim" for o in digest_options)
    assert wf["digest_gpu_roles"] == ["agent"]


def test_github_control_character_json_parses() -> None:
    bus = _Bus()
    raw = '{"card_summary": "ok", "journal_title": "t", "journal_body": "line one\nline two\ttabbed", "pr_refs": []}'

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            return _github_fetch_result(_forty_prs()[:1])
        return _Result(final_text=raw)

    result = _run(_req("github_compactor_pass"), fake, bus)
    assert result.ok is True
    assert bus.journal_bodies() == ["line one\nline two\ttabbed"]
    assert len(result.metadata["workflow"]["digest_attempts"]) == 1


def test_github_all_attempts_fail_raises_without_journal() -> None:
    bus = _Bus()

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            return _github_fetch_result(_forty_prs()[:1])
        return _Result(final_text="")

    with pytest.raises(Exception) as exc:
        _run(_req("github_compactor_pass"), fake, bus)
    assert "empty_completion" in str(exc.value)
    assert bus.journal_bodies() == []


def test_github_merge_failure_concatenates_chunk_digests() -> None:
    bus = _Bus()

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            return _github_fetch_result(_forty_prs())
        gi = req.context.metadata["github_compactor_input"]
        if "partial_digests" in gi:
            return _Result(final_text="not json")
        refs = [f"#{item['number']}" for item in gi["items"]]
        return _Result(final_text=json.dumps({"card_summary": "part", "journal_title": "p", "journal_body": " ".join(refs), "pr_refs": refs}))

    result = _run(_req("github_compactor_pass"), fake, bus)
    wf = result.metadata["workflow"]
    assert wf["digest_merge_mode"] == "concatenated"
    body = bus.journal_bodies()[0]
    assert all(f"#{1000 + i}" in body for i in range(40))


def test_github_scheduled_run_uses_denver_calendar_day(monkeypatch) -> None:
    bus = _Bus()
    fetch_args: dict = {}
    fixed_now = datetime(2026, 9, 29, 12, 10, tzinfo=timezone.utc)  # 06:10 MDT

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is None else fixed_now.astimezone(tz)

    monkeypatch.setattr(wr, "datetime", _FixedDatetime)
    items = _forty_prs()[:3]
    items[0]["merged_at"] = "2026-09-28T05:59:00Z"  # 23:59 MDT on 09-27: outside
    items[1]["merged_at"] = "2026-09-28T06:00:00Z"  # 00:00 MDT on 09-28: inside
    items[2]["merged_at"] = "2026-09-29T05:59:59Z"  # 23:59:59 MDT on 09-28: inside

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            fetch_args.update(req.context.metadata["skill_args"])
            return _github_fetch_result(items)
        gi = req.context.metadata["github_compactor_input"]
        refs = [f"#{item['number']}" for item in gi["items"]]
        return _Result(final_text=json.dumps({"card_summary": "c", "journal_title": "t", "journal_body": " ".join(refs), "pr_refs": refs}))

    result = _run(_req("github_compactor_pass", scheduled_dispatch={"source": "orion-actions"}), fake, bus)
    wf = result.metadata["workflow"]
    assert fetch_args["window_start_utc"] == "2026-09-28T06:00:00+00:00"
    assert fetch_args["window_end_utc"].startswith("2026-09-29T05:59:59")
    assert wf["window_mode"] == "day"
    assert wf["window_label"] == "2026-09-28"
    assert wf["merged_pr_count"] == 2
    assert bus.journal_bodies() == ["#1001 #1002"]


# ---------------------------------------------------------------- chat

def _chat_window(n: int) -> dict:
    turns = [
        {
            "created_at": f"2026-09-28T{6 + i // 60:02d}:{i % 60:02d}:00+00:00",
            "correlation_id": f"corr-{i}",
            "prompt": f"question {i} " + "about the substrate " * 60,
            "response": f"answer {i} " + "with a long explanation " * 120,
        }
        for i in range(n)
    ]
    return {
        "window_start_utc": "2026-09-28T06:00:00+00:00",
        "window_end_utc": "2026-09-29T05:59:59+00:00",
        "turn_count": n,
        "turns": turns,
        "transcript_text": "non-empty",
        "selection_strategy": "time_bound_recent_n",
    }


def test_chat_120_turns_all_covered_and_body_untrimmed() -> None:
    bus = _Bus()
    inputs: list[dict] = []
    skill_args: dict = {}
    long_body = "Whole day. " + ("topic covered; " * 2000)  # ~30k chars, old cap 4000

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.chat.discussion_window.v1":
            skill_args.update(req.context.metadata["skill_args"])
            window = _chat_window(120)
            return _Result(final_text=json.dumps(window), metadata={"skill_result": window})
        ci = req.context.metadata["chat_history_compactor_input"]
        inputs.append(ci)
        if "partial_digests" in ci:
            refs = [r for part in ci["partial_digests"] for r in part["turn_refs"]]
            return _Result(final_text=json.dumps({"card_summary": "day", "journal_title": "Chat day", "journal_body": long_body, "turn_refs": refs}))
        refs = [t["correlation_id"] for t in ci["turns"]]
        return _Result(final_text=json.dumps({"card_summary": "part", "journal_title": "p", "journal_body": " ".join(refs), "turn_refs": refs}))

    result = _run(_req("chat_history_compactor_pass", scheduled_dispatch={"source": "orion-actions"}), fake, bus)
    wf = result.metadata["workflow"]
    chunk_inputs = [ci for ci in inputs if "turns" in ci]
    covered = sorted(int(t["correlation_id"].split("-")[1]) for ci in chunk_inputs for t in ci["turns"])
    assert covered == list(range(120))
    assert len(chunk_inputs) > 1
    assert skill_args["max_turns"] > 200
    assert wf["total_count"] == wf["covered_count"] == 120
    assert wf["input_truncated"] is False
    assert wf["digest_merge_mode"] == "llm_merge"
    assert bus.journal_bodies() == [long_body]


# ---------------------------------------------------------------- review follow-ups

def _fixed_now(monkeypatch, now: datetime) -> None:
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is None else now.astimezone(tz)

    monkeypatch.setattr(wr, "datetime", _FixedDatetime)


def _simple_digest(gi: dict) -> _Result:
    refs = [f"#{item['number']}" for item in gi.get("items", [])]
    return _Result(final_text=json.dumps({"card_summary": "c", "journal_title": "t", "journal_body": " ".join(refs) or "b", "pr_refs": refs}))


def test_github_day_mode_widens_lookback_and_flags_old_exec(monkeypatch) -> None:
    """New orch + old exec: the old exec ignores window args and fetches now-N days."""
    _fixed_now(monkeypatch, datetime(2026, 9, 29, 12, 10, tzinfo=timezone.utc))
    bus = _Bus()
    fetch_args: dict = {}

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            fetch_args.update(req.context.metadata["skill_args"])
            return _github_fetch_result(_forty_prs()[:2])  # no window_mode echo = old exec
        return _simple_digest(req.context.metadata["github_compactor_input"])

    result = _run(_req("github_compactor_pass", scheduled_dispatch={"s": 1}), fake, bus)
    wf = result.metadata["workflow"]
    # Day starts 2026-09-28T06:00Z, ~30h before 12:10Z: a 1-day rolling fetch would miss it.
    assert fetch_args["lookback_days"] == 2
    assert wf["fetch_window_unconfirmed"] is True
    assert wf["input_truncated"] is True


def test_github_new_exec_window_echo_and_page_cap(monkeypatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 9, 29, 12, 10, tzinfo=timezone.utc))

    async def fake_factory(page_cap_hit):
        async def fake(*args, **kwargs):
            req = kwargs["client_request"]
            if req.verb == "skills.repo.github_recent_prs.v1":
                return _github_fetch_result(_forty_prs()[:2], window_mode="window", page_cap_hit=page_cap_hit)
            return _simple_digest(req.context.metadata["github_compactor_input"])
        return fake

    ok = _run(_req("github_compactor_pass", scheduled_dispatch={"s": 1}), asyncio.run(fake_factory(False)), _Bus())
    assert ok.metadata["workflow"]["input_truncated"] is False
    assert ok.metadata["workflow"]["fetch_window_unconfirmed"] is False
    capped = _run(_req("github_compactor_pass", scheduled_dispatch={"s": 1}), asyncio.run(fake_factory(True)), _Bus())
    assert capped.metadata["workflow"]["input_truncated"] is True
    assert capped.metadata["workflow"]["github_page_cap_hit"] is True


def test_github_merge_that_drops_refs_falls_back_to_concatenation() -> None:
    bus = _Bus()

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            return _github_fetch_result(_forty_prs())
        gi = req.context.metadata["github_compactor_input"]
        if "partial_digests" in gi:
            return _Result(final_text=json.dumps({"card_summary": "m", "journal_title": "t", "journal_body": "only #1000", "pr_refs": ["#1000"]}))
        return _simple_digest(gi)

    wf = _run(_req("github_compactor_pass"), fake, bus).metadata["workflow"]
    assert wf["digest_merge_mode"] == "concatenated"
    assert wf["digest_merge_skipped_reason"].startswith("merge_dropped_refs:")
    assert all(f"#{1000 + i}" in bus.journal_bodies()[0] for i in range(40))


def test_github_merge_input_over_budget_skips_merge_call(monkeypatch) -> None:
    monkeypatch.setattr(map_reduce, "DIGEST_INPUT_CHAR_BUDGET", 20_000)
    bus = _Bus()
    merge_calls = []

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            return _github_fetch_result(_forty_prs())
        gi = req.context.metadata["github_compactor_input"]
        if "partial_digests" in gi:
            merge_calls.append(1)
        refs = [f"#{item['number']}" for item in gi.get("items", [])]
        body = " ".join(refs) + " " + ("x" * 5000)
        return _Result(final_text=json.dumps({"card_summary": "c", "journal_title": "t", "journal_body": body, "pr_refs": refs}))

    wf = _run(_req("github_compactor_pass"), fake, bus).metadata["workflow"]
    assert merge_calls == []
    assert wf["digest_merge_mode"] == "concatenated"
    assert wf["digest_merge_skipped_reason"] == "merge_input_over_budget"
