"""The day dream's memories can pass the grounding gate (PR dream-day-citable-facts, #3739).

Live 2026-11-18..22: every night wrote 5 memories and the gate refused all 5, rightly: the cited records did not
carry the facts. What must hold now:

* a day's totals have a citable home, ``dream_day_read:<day>``, which resolves deterministically to THIS read's
  count lines (the totals file `dig dream-day` left with the receipt's read_at), the judge sees them, and the row
  keeps them so a stored night resolves the same way;
* the writer's prompt carries the facts rule, the totals ref and the answer shape before the steps;
* a turn the motor cut short (the real 11-19 replies) is a named failed attempt, never parsed; a real answer is
  found fenced or bare, with prose or another code block after it, and a printed read receipt is never an answer;
* a retry is skipped for time only when less than DREAM_DAY_MIN_ATTEMPT_SEC of the night is left.

(The per-kind detail SQL and the reminiscence receipt are checked against a real Postgres in
test_dream_day_postgres.py.)
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from app.dream_day_graph import (
    DreamDayDeps, build_dream_day_graph, build_prompt, check_reply, retry_attempt_sec, stored_read,
)
from app.dream_day_receipts import receipt_hop_id
from app.harness_step import HarnessStepFailed
from orion.memory.dream_day import (
    GROUNDING_RECORDS_CHARS, TOTALS_CHARS, grounding_prompt, looks_like_answer, parse_answer, totals_ref,
    totals_resolved, totals_text,
)
from orion.schemas.dream_day import (
    DREAM_DAY_MIN_ATTEMPT_SEC, DREAM_DAY_WORKFLOW, DreamDayBriefV1, DreamDayTotalsV1, split_ref,
)

from test_dream_day import (  # noqa: E402  (same-dir test module, like the other graph tests)
    ACT, CFG, CYCLE, DAY, EXP, RESOLVED, RUN_ID, T0, World, answer_json, material,
)

ROUNDUP = [
    "organs 2026-11-17 (Denver; 2026-11-17T07:00Z to 2026-11-18T07:00Z): 16 organs, 2507 experiences",
    "body: 226 experiences (body_binding 131, gpu_wait 75, arousal_transition 20)",
    '  "durable-runs:x waited 8177057 ms for a GPU [granted]" [experience:gpu_wait:c3f7] salience 0.97',
    '  cited from body: "durable-runs:y waited 7831646 ms for a GPU [granted]" [experience:gpu_wait:d681]',
    "  write-only: 0 of its 107 experiences cited by another organ",
    "acts by actors no organ claims: hub.promise 3",
]
TOTALS_REF = totals_ref(DAY)


# --------------------------------------------------------------------------- the totals ref


def test_totals_text_keeps_count_lines_drops_quotes_and_caps():
    text = totals_text(ROUNDUP, ["digs that day: 79 dig command(s), 3 in chat, 76 in background runs"])
    lines = text.split("\n")
    assert "body: 226 experiences (body_binding 131, gpu_wait 75, arousal_transition 20)" in lines
    assert "acts by actors no organ claims: hub.promise 3" in lines and lines[-1].startswith("digs that day: 79")
    assert not any("waited" in ln for ln in lines)     # a quoted record or cite is citable on its own
    assert totals_text(ROUNDUP) == totals_text(list(ROUNDUP))   # deterministic
    big = totals_text([f"organ {i}: {i} experiences " + "x" * 80 for i in range(500)])
    assert len(big) <= TOTALS_CHARS + 30 and big.endswith("(more count lines cut)")


def test_the_totals_ref_is_a_ref_kind_and_resolves_only_for_its_own_night():
    assert split_ref(f"[{TOTALS_REF}]") == ("dream_day_read", "2026-11-17")
    totals = {"day": "2026-11-17", "read_at": "2026-11-18T08:40:00Z", "text": "body: 226 experiences (gpu_wait 75)"}
    full = totals_resolved(DAY, totals, full=True)
    assert full == {TOTALS_REF: "dig dream-day 2026-11-17 (my read of that day, at 2026-11-18T08:40:00Z) "
                                "body: 226 experiences (gpu_wait 75)"}
    short = totals_resolved(DAY, totals, full=False)
    assert "gpu_wait 75" not in short[TOTALS_REF]     # the stored evidence quote stays short
    assert totals_resolved(DAY, {**totals, "day": "2026-11-16"}, full=True) == {}
    assert totals_resolved(DAY, None, full=True) == {} and totals_resolved(DAY, {**totals, "text": " "}, full=True) == {}


def test_stored_read_keeps_the_totals_for_a_stored_night():
    receipt = {"day": "2026-11-17", "read_at": "2026-11-18T08:40:00Z", "unread": {}}
    assert stored_read({"read": receipt}) == receipt
    totals = {"day": "2026-11-17", "read_at": "2026-11-18T08:40:00Z", "text": "gpu_wait 75"}
    assert stored_read({"read": receipt, "read_totals": totals}) == {
        **receipt, "totals": "gpu_wait 75", "totals_read_at": "2026-11-18T08:40:00Z"}


class TotalsWorld(World):
    """test_dream_day's World, plus the totals file the dig leaves beside the receipt."""

    def __init__(self, *a, totals_text_: str | None = "body: 226 experiences (gpu_wait 75)", totals_at=None, **kw):
        super().__init__(*a, **kw)
        self.totals_text, self.totals_at = totals_text_, totals_at

    async def read_totals(self, day):
        if self.totals_text is None:
            return None
        return DreamDayTotalsV1(day=day, read_at=self.totals_at or self.receipt_at or self.now, text=self.totals_text)

    def graph(self, saver):
        from app.admitted_graph import AdmissionDeps
        deps = DreamDayDeps(run_turn=self.turn, gather=self.gather, read_receipt=self.read_receipt,
                            resolve_refs=self.resolve, known_referents=self.referents, persist=self.persist,
                            record_cites=self.record_cites, judge=self._judge if self.judge else None,
                            read_totals=self.read_totals)
        return build_dream_day_graph(deps, AdmissionDeps(self.register, self.lease, self.execute, self.release,
                                                         self.event, now=lambda: self.now,
                                                         max_attempts=self.max_attempts, retry_base_seconds=10.0),
                                     saver)


TOTALS_MEMORY = {"statement": "My body queued hard that day: 75 gpu waits.", "purpose": "happened",
                 "refs": [TOTALS_REF]}


def initial(timeout_sec=900, deadline=None):
    admission = {"resource": "llm.route.memory_distill"}
    if deadline is not None:
        admission["deadline_at"] = deadline.isoformat()
    return {"run_id": RUN_ID, "correlation_id": "trace-dd", "workflow": DREAM_DAY_WORKFLOW, "attempt": 0,
            "requested_at": (T0 - timedelta(minutes=1)).isoformat(), "admission": admission,
            "brief": {"dream_day": "dream.day", "day": DAY.isoformat(), "timeout_sec": timeout_sec,
                      "llm_route": "memory_distill"}}


async def run(world, saver, state=None):
    pytest.importorskip("langgraph")
    from langgraph.types import Command
    await asyncio.wait_for(world.graph(saver).ainvoke(state or initial(), CFG), 2)
    world.granted = True
    return await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume={}), CFG), 5)


def _judge_says(*answers):
    queue = list(answers)

    async def judge(prompt):
        return {"text": json.dumps(queue.pop(0)), "finish_reason": "stop"}
    return judge


def test_a_memory_citing_the_nights_read_is_judged_on_its_count_lines_and_written():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world = TotalsWorld(replies=[answer_json(memories=[TOTALS_MEMORY])],
                            judge=_judge_says({"supported": True, "unsupported_claims": []}))
        result = await run(world, InMemorySaver())
        assert result["status"] == "completed" and result["produced"] is True
        prompt = world.judged[0][0]
        assert f"[{TOTALS_REF}] dig dream-day 2026-11-17 (my read of that day" in prompt
        assert "body: 226 experiences (gpu_wait 75)" in prompt   # the judge sees the count lines
        fields = world.stored["fields"]
        assert fields["read"]["totals"] == "body: 226 experiences (gpu_wait 75)"
        quote = world.stored["result"].memories[0].evidence[0]
        assert quote.source_id == TOTALS_REF and "gpu_wait 75" not in quote.quote   # stored quote stays short
    asyncio.run(scenario())


def test_another_reads_totals_never_back_this_runs_memory():
    """A totals file whose read_at is not this receipt's (an older or a later read) is not this attempt's: the ref
    does not resolve, the memory is dropped like any unresolved ref, nothing is written."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world = TotalsWorld(replies=[answer_json(memories=[TOTALS_MEMORY])], totals_at=T0 - timedelta(hours=3),
                            judge=_judge_says())
        result = await run(world, InMemorySaver())
        assert result["status"] == "completed" and result["produced"] is False
        assert world.judged == [] and result["summary"]["items_dropped"] == 1
        assert "totals" not in (world.stored["fields"]["read"] or {})
    asyncio.run(scenario())


# --------------------------------------------------------------------------- the writer's prompt


def test_the_prompt_carries_the_facts_rule_the_totals_ref_and_the_shape_first():
    brief = DreamDayBriefV1(dream_day="dream.day", day=DAY, timeout_sec=1200, llm_route="memory_distill")
    prompt = build_prompt(brief, material())
    assert "must be stated in a record that memory cites" in prompt
    assert "A judge reads ONLY the cited records" in prompt and "refused" in prompt
    assert "prefer fewer, smaller memories" in prompt
    assert f'cite "{TOTALS_REF}"' in prompt and "One record (one gpu wait, one run) never shows a total" in prompt
    # The answer shape comes before the steps and the material; the last line repeats where the answer goes.
    assert prompt.index("```json") < prompt.index("Run `python -P -m orion.dig dream-day 2026-11-17` FIRST")
    assert prompt.index("```json") < prompt.index("CONCERNS PRIME HOLDS NOW")
    assert prompt.rstrip().splitlines()[-1].startswith("Remember: end your reply with the ```json answer block")
    assert "at most 3 more commands" in prompt and "a turn that runs out of time saves nothing" in prompt
    assert "Attempt" not in prompt.splitlines()[0]
    retry = build_prompt(brief, material(), attempt=1)
    assert retry.startswith("Attempt 2: your last attempt ran out of time before it wrote the answer block")
    assert len(prompt) < 6000   # short: the material is the bulk


# --------------------------------------------------------------------------- answer-block failures

# The real 11-19 replies (turns 676dd36b-..., 04471fcf-...), trimmed: the motor stopped both mid-dig (fcc_timeout
# after 60 and 30 steps) and handed back its cut-short draft; the second contains the read receipt it cat'ed.
REAL_1119_REPLY = (
    "[Cut short - not a finished answer.]\n"
    "This turn was stopped after 60 steps because its time budget ran out (fcc_timeout), before Orion wrote a "
    "conclusion. What follows is what the turn actually recorded (19 tool results), in order. None of it has been "
    "synthesized into an answer.\n\n(3 earlier entries omitted for length.)\n"
    '- Bash {"command": "cat /mnt/orion-fcc/self/dream_day/2026-11-19.json", "description": "Read the dream-day read '
    'receipt"} returned: { "day": "2026-11-19", "read_at": "2026-11-20T08:44:01Z", "roundup_lines": 86 }\n'
    '- mcp__orion-introspect__curiosity {"run_id": "1f0bfc94d63f"} returned: {"ok": true, "operation": "curiosity"}\n')


def test_a_cut_short_reply_is_its_own_failed_attempt_never_parsed():
    record = {"debug": {"harness_grounding_status": "fcc_timeout", "harness_step_count": 60}}
    with pytest.raises(HarnessStepFailed, match="dream_day_cut_short: fcc_timeout after 60 steps"):
        check_reply(REAL_1119_REPLY, record)
    with pytest.raises(HarnessStepFailed, match="dream_day_cut_short"):
        check_reply(REAL_1119_REPLY)          # the marker alone is enough
    # Even parsed raw, the receipt object the turn printed is never read as an answer.
    assert parse_answer(REAL_1119_REPLY) is None
    assert not looks_like_answer({"day": "2026-11-19", "read_at": "2026-11-20T08:44:01Z"})


@pytest.mark.parametrize("reply", [
    # the live shape (11-20, 11-22): prose, then one fenced json block at the very end
    "I read the whole day.\n\n```json\n{body}\n```\n",
    # a fence with no language tag, prose after it
    "Here it is:\n```\n{body}\n```\nThat is what I keep.",
    # the object on the fence's own line, then a second code block of another language
    "```json {body}```\n\nI ran:\n```bash\npython -P -m orion.dig act act_1\n```",
    # an unclosed brace in the prose before a bare answer
    "The set {a, b of things I saw. {body}",
])
def test_the_answer_is_found_fenced_or_bare_with_anything_after_it(reply):
    body = json.loads(answer_json().split("```json\n", 1)[1].rsplit("\n```", 1)[0])
    parsed = check_reply(reply.replace("{body}", json.dumps(body)), {"debug": {"harness_step_count": 40}})
    assert len(parsed.answer.memories) == 1 and parsed.answer.memories[0].refs == [ACT, EXP]
    assert parsed.answer.day.refs == [CYCLE]


def test_the_last_answer_wins_and_a_reply_with_none_is_no_answer():
    first = json.dumps({"memories": [], "plan": [], "day": {"text": "a first draft of the day", "refs": []}})
    last = json.dumps({"memories": [], "plan": [], "day": {"text": "the final paragraph of the day", "refs": []}})
    assert check_reply(f"draft {first} then final {last} done").answer.day.text == "the final paragraph of the day"
    with pytest.raises(HarnessStepFailed, match="dream_day_no_answer"):
        check_reply('I looked at {"day": "2026-11-19", "read_at": "x"} and wrote nothing.')


# --------------------------------------------------------------------------- the retry's time budget


def test_retry_attempt_sec_is_the_least_useful_attempt():
    assert retry_attempt_sec(1200) == DREAM_DAY_MIN_ATTEMPT_SEC == 900.0
    assert retry_attempt_sec(600) == 600.0


def test_a_retry_runs_while_the_night_still_has_a_useful_attempt_left():
    """Brief 1200 s, night deadline 1000 s after the failed attempt: the retry fits a useful attempt (>= 900 s), so it
    is scheduled, not skipped as before (the old check wanted a whole 1200 s brief attempt)."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario(left_sec):
        world = TotalsWorld(replies=[REAL_1119_REPLY])
        state = initial(timeout_sec=1200, deadline=T0 + timedelta(seconds=left_sec))
        return await run(world, InMemorySaver(), state)

    room = asyncio.run(scenario(1000 + 30))   # + the retry's 10 s backoff and slack
    assert room["status"] == "retrying" and "dream_day_cut_short" in room["last_error"]
    late = asyncio.run(scenario(600))
    assert late["status"] == "failed" and late["last_error"].startswith("retry_skipped_insufficient_time")


def test_a_retry_tells_the_turn_why_the_last_attempt_saved_nothing():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    async def scenario():
        world, saver = TotalsWorld(replies=[REAL_1119_REPLY, answer_json()]), InMemorySaver()
        result = await run(world, saver)
        assert result["status"] == "retrying"
        snap = await world.graph(saver).aget_state(CFG)
        world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
        world.granted = True
        result = await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume=True), CFG), 5)
        assert result["status"] == "completed"
        assert not world.turns[0].prompt.startswith("Attempt")
        assert world.turns[1].prompt.startswith("Attempt 2: your last attempt ran out of time")
    asyncio.run(scenario())


def test_the_tick_submits_a_new_run_with_a_useful_attempt_left():
    from test_dream_day import TickPool, tick_settings
    from app.dream_day_poll import DreamDayTicker

    failed = [{"run_id": RUN_ID, "terminal": "failed", "updated_at": T0 - timedelta(hours=1)}]
    sent = []

    async def submit(req):
        sent.append(req)
    # Window ends 10:00Z; 950 s left: under the 1200 s brief, over the 900 s least useful attempt.
    now = T0.replace(hour=9, minute=44, second=10)
    t = DreamDayTicker(settings=tick_settings(dream_day_turn_timeout_sec=1200.0), pool=TickPool(runs=failed),
                       submit=submit, now=lambda: now)
    result = asyncio.run(t.tick_once())
    assert result.decision == "submitted" and sent[0].brief.timeout_sec == 1200.0


# --------------------------------------------------------------------------- the judge's page and #3739


def test_the_reminiscence_receipt_names_its_hop():
    assert receipt_hop_id("reply: reminisce what_it_was saved rm:2026-11-21:what_it_was:ffc6f3f855") \
        == "rm:2026-11-21:what_it_was:ffc6f3f855"
    assert receipt_hop_id("reminisce otherwise saved rm:2026-11-17:otherwise:5ffeb83c1e; the week 2025-10-14..") \
        == "rm:2026-11-17:otherwise:5ffeb83c1e"
    assert receipt_hop_id("reply: I followed the question I queued for myself") is None
    assert receipt_hop_id("reminisce cross-read done") is None


def test_the_judges_page_is_capped_per_memory():
    refs = [f"act:act_{i}" for i in range(6)]
    resolved = {r: "§" * 5000 for r in refs}
    prompt = grounding_prompt("A statement.", refs, resolved)
    assert prompt.count("§") <= GROUNDING_RECORDS_CHARS
    assert "(cut: the judge's page is full)" in prompt
    small = grounding_prompt("A statement.", [ACT], RESOLVED)
    assert f"[{ACT}] {RESOLVED[ACT]}" in small and "(cut" not in small
