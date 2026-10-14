"""Replay of reading seed c31227dc Stage 2 (2026-10-14): a huge source file, no draft.

What happened (live records): Stage 1 read the GitHub *blob* page of a 222 KB C file
and kept only page chrome (5,544-char digest, "zero lines of C"). Stage 2's ~21k-char
prompt was fine; the model then fetched the raw file itself, and asked WebFetch for
"characters 100000 to 200000" again and again. WebFetch only ever reads a page's
first ~100k characters, so every repeat re-ran the same ~42k-token summarizer prompt
through the single agent-lane slot (5-19 min each) until the 3500 s attempt cut the
turn with draft_len=0. Attempts 2 and 3 were sent the byte-identical prompt.

This replay answers, from frozen records only (no network, no DB):
  1. prompt: stored Stage 2 prompt size per attempt vs what the new builder sends
     (one WebFetch-limit line; plus a no-fetch RETRY line after a timeout);
  2. harness: which of the real turn's tool calls the reading-turn hook rules
     (refetch + tool window, orion/fcc/repeat_failure_breaker.py) block, and how
     many seconds of single-slot lane wait those calls actually cost.

Run: PYTHONPATH=<worktree>:<worktree>/services/orion-hub python services/orion-hub/evals/run_reading_huge_page_replay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
SEED_FIXTURE = _HERE / "fixtures" / "reading_huge_page_c31227dc_2026-10-14.json"
TURN_FIXTURE = _REPO / "orion" / "fcc" / "tests" / "fixtures" / "fcc_reading_refetch_c31227dc.jsonl"


def _turn():
    lines = TURN_FIXTURE.read_text().splitlines()
    return json.loads(lines[0])["_meta"], [json.loads(line) for line in lines[1:]]


def replay_harness(*, refetch_limit: int, fraction: float, budget: float) -> dict:
    from orion.fcc import repeat_failure_breaker as rfb

    meta, entries = _turn()
    ends = {}
    starts = {}
    for e in entries:
        for b in e["message"]["content"]:
            if b["type"] == "tool_use":
                starts[b["id"]] = (e["_t"], b["name"], b["input"].get("url") or b["input"].get("query"))
            elif b["type"] == "tool_result":
                ends[b["tool_use_id"]] = e["_t"]
    env = {"ORION_TURN_BUDGET_SEC": str(budget), "ORION_TURN_DEADLINE_EPOCH": str(budget)}
    calls = []
    for i, e in enumerate(entries):
        for b in e["message"]["content"]:
            if b["type"] != "tool_use":
                continue
            history = rfb.CallHistory().feed_all(entries[:i])
            rule = None
            if rfb.tool_window_state(env, fraction=fraction, now=float(e["_t"])) is not None:
                rule = "tool_window"
            elif rfb.decide_refetch(history, b["name"], b["input"], limit=refetch_limit):
                rule = "refetch"
            st, name, target = starts[b["id"]]
            calls.append({"id": b["id"], "tool": name, "target": target, "start_s": st,
                          "real_wait_s": ends[b["id"]] - st, "blocked_by": rule})
    allowed = [c for c in calls if not c["blocked_by"]]
    return {
        "budget_s": budget,
        "calls": calls,
        "blocked": [c["id"] for c in calls if c["blocked_by"]],
        "blocked_calls_wait_s_summed_overlapping": sum(c["real_wait_s"] for c in calls if c["blocked_by"]),
        # Upper bound: in the real turn these waited behind the calls that would now be blocked.
        "last_allowed_result_s": max(ends[c["id"]] for c in allowed) if allowed else 0,
        "turn_killed_at_s": meta["killed_at_s"],
    }


def replay_prompt() -> dict:
    from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1
    from orion.world_pulse_read.fetch_text import WEBFETCH_LIMIT_NOTE
    from scripts.world_pulse_read_stage2 import _build_stage2_prompt

    seed = json.loads(SEED_FIXTURE.read_text())
    handoff = WorldPulseReadHandoffV1.model_validate(seed["handoff_json"])
    first = _build_stage2_prompt(handoff, "tr-replay")
    retry = _build_stage2_prompt(handoff, "tr-replay", prior_error="turn_exception:TimeoutError: ")
    stored = {p["attempt"]: p["prompt_chars"] for p in seed["stage2_prompts"]}
    return {
        "retained_text_chars": seed["retained_chars"],
        "stored_prompt_chars_by_attempt": stored,
        "stored_attempts_2_and_3_identical": stored.get(2) == stored.get(3),
        "new_note_chars": len(WEBFETCH_LIMIT_NOTE),
        "new_retry_note_chars": len(retry) - len(first),
        # Offline rebuild omits the claim block (needs the live graph store), so
        # this is the handoff-only prompt; the delta above is exact.
        "rebuilt_without_claim_block_chars": len(first),
        "retry_prompt_differs": retry != first,
    }


def main() -> int:
    meta, _ = _turn()
    stance_s = meta["harness_started_after_attempt_s"]
    after_budget = meta["durable_attempt_sec"] - stance_s - 10  # listener stamp + orchestrator slack
    report = {
        "seed": meta["seed"],
        "prompt": replay_prompt(),
        "harness_before": replay_harness(refetch_limit=0, fraction=0.0, budget=7200),
        "harness_after": replay_harness(refetch_limit=1, fraction=0.5, budget=after_budget),
    }
    json.dump(report, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
