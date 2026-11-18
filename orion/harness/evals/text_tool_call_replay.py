"""Replay real turns whose final draft was a tool call written as text (Strata, 2026-10-31 on).

For each draft: would the motor now detect it for the one resume (and does the note say how to
load the tool), and if the resume fails too, what does the runner hand finalize? Before the fix the motor resumed only
reading turns, so an ordinary turn's text call always went to finalize.

    python -m orion.harness.evals.text_tool_call_replay                 # live, via docker exec psql
    python -m orion.harness.evals.text_tool_call_replay --json FILE     # [{"correlation_id", "draft_text"}]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections import Counter
from typing import Iterable

from orion.harness.cut_short import TEXT_TOOL_CALL_CODE
from orion.harness.fcc_motor import text_tool_call, text_tool_call_note

SQL = (
    "select coalesce(json_agg(json_build_object('correlation_id', correlation_id, 'created_at', created_at, "
    "'draft_text', run_artifact->>'draft_text') order by created_at), '[]') from harness_turn_trace "
    "where created_at > '{since}' and run_artifact->>'draft_text' like '%<function=%'"
)


def load_live(since: str) -> list[dict]:
    out = subprocess.run(
        ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "orion_readonly", "-d", "conjourney", "-tAc",
         SQL.format(since=since)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


async def _runner_outcome(draft: str) -> tuple[str, str | None, str]:
    """Run the harness runner on the live shape of a turn that ends on the call -- the call as an
    assistant text step, then the final -- and report what reaches finalize."""
    from unittest.mock import AsyncMock

    from orion.harness.runner import HarnessRunner
    from orion.harness.tests.fixtures import make_thought
    from orion.schemas.cognition.answer_contract import AnswerContract
    from orion.schemas.context_exec import ContextExecPermissionV1
    from orion.schemas.harness_finalize import HarnessRunRequestV1

    async def fcc(**_: object):
        raw = {"type": "assistant", "message": {"content": [{"type": "text", "text": draft}]}}
        yield {"type": "step", "step": {"type": "assistant", "raw": raw}}
        yield {"type": "final", "llm_response": draft, "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(correlation_id="replay", thought_event=make_thought(), user_message="replay",
                                  permissions=ContextExecPermissionV1(), answer_contract=AnswerContract())
    result = await HarnessRunner(AsyncMock(), fcc_runner=fcc).run(request)
    return result.draft_text, result.cut_short_reason, result.compliance_verdict


def replay(rows: Iterable[dict]) -> dict:
    """Detection, the resume note, and what the runner hands finalize if the resume fails too.

    Not covered: whether the model completes the step after the note (needs a live turn)."""
    rows = list(rows)
    detected, notes_ok, raw_call_reaches_finalize, outcomes, tools = 0, 0, 0, Counter(), Counter()
    misses: list[str] = []
    for r in rows:
        tool = text_tool_call(r["draft_text"])
        if not tool:
            misses.append(str(r.get("correlation_id")))
            continue
        detected += 1
        tools[tool.rsplit("__", 1)[-1]] += 1
        if f"select:{tool}" in text_tool_call_note(tool) or not tool.startswith("mcp__"):
            notes_ok += 1
        draft, cut_reason, verdict = asyncio.run(_runner_outcome(r["draft_text"]))
        if "<tool_call>" in draft or (draft and cut_reason != TEXT_TOOL_CALL_CODE):
            raw_call_reaches_finalize += 1
        outcomes["cut_short_draft" if draft else f"motor_{verdict}"] += 1
    return {
        "turns": len(rows),
        "detected_for_resume": detected,
        "note_names_toolsearch_select": notes_ok,
        "runner_if_resume_fails": dict(outcomes),
        "raw_call_or_repairable_draft_reaches_finalize": raw_call_reaches_finalize,
        "tools": dict(tools),
        "not_detected": misses,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="file of rows instead of the live DB")
    ap.add_argument("--since", default="2026-10-31 08:00")
    args = ap.parse_args(argv)
    rows = json.load(open(args.json)) if args.json else load_live(args.since)
    report = replay(rows)
    print(json.dumps(report, indent=1))
    ok = report["turns"] and not report["not_detected"] and not report["raw_call_or_repairable_draft_reaches_finalize"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
