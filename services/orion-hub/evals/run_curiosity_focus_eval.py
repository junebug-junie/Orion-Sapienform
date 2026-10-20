#!/usr/bin/env python3
"""Curiosity focus eval: does a focus Orion queues from chat reach exactly one investigate run,
and does Orion's own `curiosity` read tool then say truthfully what that run did with it?

Round trip through the real code at every step, over the in-memory focus table
(orion/curiosity/tests/focus_fixtures.py, whose SQL semantics are proven against Postgres in
orion/curiosity/tests/test_focus_lifecycle.py):

1. Queue: Hub's queue_curiosity_focus responder over a bus envelope from a chat turn. Orion then
   tries to replace a focus Juniper asked for and must be refused with nothing written; a
   Juniper-origin call replaces it and the reply names what it replaced.
2. Take: CuriosityInvestigation._take_focus for run A on a readable graph; the real
   build_kickoff_prompt shows the question first and asks for focus_disposition with run A's id.
   A second run takes nothing. The typed copy Hub hands the governor (CuriosityFocusAnchorV1)
   renders into the --append-system-prompt focus anchor, so compaction cannot lose it.
3. Outcome: run A wrote `focus_disposition: "set_aside: ..."` on its :TurnOutcome;
   CuriosityInvestigation._stamp_focus reads it back (read-only) and stamps it.
4. Read back: the `curiosity` responder with kind=focus says "taken by run A ... set aside".
5. Unknown: a second focus taken by run B whose run wrote no disposition reads "unknown", never
   "explored".

Deterministic and offline: no DB, LLM, bus, graph or network.

    python services/orion-hub/evals/run_curiosity_focus_eval.py

Exit 0 = every check passed, 1 = at least one failed.
"""
from __future__ import annotations

import asyncio
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]
warnings.filterwarnings("ignore", message=r"Field \"model_", category=UserWarning)

RUN_A = "a1a1a1a1a1a1"
RUN_B = "b2b2b2b2b2b2"


def _paths() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_HUB_ROOT)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_HUB_ROOT))


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


class _Reader:
    """Orion's graph after run A: one :TurnOutcome carrying a disposition (none for run B)."""

    def __init__(self, dispositions):
        self.dispositions = dispositions

    def query(self, cypher):
        for run_id, value in self.dispositions.items():
            if f"t.run_id = '{run_id}'" in cypher and "focus_disposition" in cypher:
                return [{"focus_disposition": value}]
        return []


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- one linear scenario
    _paths()
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.curiosity import focus as cf
    from orion.curiosity.kickoff_prompt import build_kickoff_prompt
    from orion.curiosity.study_material import StudyMaterial
    from orion.curiosity.tests.focus_fixtures import FocusPool, FocusTable
    from orion.curiosity.worldview import WorldviewSnapshot
    from orion.introspect.transport import REQUEST_KIND, RESULT_PREFIX
    from orion.schemas.introspect import IntrospectResultV1, IntrospectToolBindingV1
    from scripts.curiosity_focus_listener import CuriosityFocusListener
    from scripts.curiosity_introspect_listener import CuriosityIntrospectListener
    from scripts.curiosity_investigation import CuriosityInvestigation

    table = FocusTable()
    pool = FocusPool(table)
    failures: list[str] = []
    chat = IntrospectToolBindingV1(invocation_context="unified_chat", parent_run_id="chat",
                                   parent_trace_id="chat-corr-eval", memory_allowed=True)

    async def _schedule():
        return {"available": True, "tz": "America/Denver", "lines": {"investigate": {
            "enabled": True, "runs_today": 7, "daily_cap": 7, "next_eligible_at": None}}}

    queue_listener = CuriosityFocusListener(pool_provider=lambda: pool, source_ref=ServiceRef(name="orion-hub"),
                                            enabled=True, schedule_provider=_schedule)
    queue_listener.bus = _Bus()

    def ask(listener, operation, args) -> IntrospectResultV1:
        corr = uuid4()
        asyncio.run(listener.handle(BaseEnvelope(
            kind=REQUEST_KIND, correlation_id=corr, source=ServiceRef(name="orion-harness-governor"),
            reply_to=f"{RESULT_PREFIX}{corr}",
            payload={"operation": operation, "binding": chat.model_dump(mode="json"), "args": args},
        )))
        return IntrospectResultV1.model_validate(listener.bus.published[-1][1].payload)

    # 1. Queue, with the Juniper rule.
    hers = ask(queue_listener, "queue_curiosity_focus",
               {"question": "Why did the fans spin up at 3am?", "why": "Juniper asked", "origin": "juniper"})
    refused = ask(queue_listener, "queue_curiosity_focus",
                  {"question": "My own idea", "why": "curious", "origin": "self"})
    if not hers.ok:
        failures.append(f"Juniper's focus refused: {hers.error}")
        print("\n".join(f"FAIL {f}" for f in failures))
        return 1
    if refused.ok or cf.JUNIPER_PROTECTED not in (refused.error or ""):
        failures.append(f"Orion replaced Juniper's focus: {refused}")
    if len(table.rows) != 1:
        failures.append(f"a refused queue wrote a row ({len(table.rows)} rows)")
    planted = ask(queue_listener, "queue_curiosity_focus",
                  {"question": "How much of the bus traffic is heartbeats?", "why": "she said: look at this instead",
                   "origin": "juniper"})
    if not planted.ok or (planted.items[0].extra.get("replaced") or {}).get("focus_id") != hers.items[0].id:
        failures.append("Juniper's second ask did not replace her first, or the reply did not name it")
    if "used up (7 of 7)" not in planted.items[0].extra["next_run"]["words"]:
        failures.append("the reply did not say the line is at today's cap")
    fid = planted.items[0].id

    # 2. Take, and the kickoff.
    loop = CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=1800.0, daily_cap=7, timeout_sec=60.0,
        session_id="eval", crystallization_sample=1, relation_sample=1, pool_provider=lambda: pool,
        source_ref=ServiceRef(name="orion-hub"),
        reader=_Reader({RUN_A: "set_aside: the bus logs for that hour were empty"}), focus_enabled=True,
    )
    taken = asyncio.run(loop._take_focus(WorldviewSnapshot(), RUN_A))
    if taken is None or taken.focus_id != fid:
        failures.append(f"run A did not take the planted focus: {taken}")
    prompt = build_kickoff_prompt(StudyMaterial(generated_at=datetime.now(timezone.utc)),
                                  view=WorldviewSnapshot(), run_id=RUN_A, focus=taken)
    for needle in (cf.FOCUS_HEADER, "How much of the bus traffic is heartbeats?", "Juniper asked you",
                   'focus_disposition: "explored"', f'run_id: "{RUN_A}"', "offer, not an order",
                   "recorded as unknown, not as explored"):
        if needle not in prompt:
            failures.append(f"kickoff lacks {needle!r}")
    if prompt.index(cf.FOCUS_HEADER) > prompt.index("LAST, IF YOU WANT EITHER OF THESE"):
        failures.append("the focus is not shown before the outcome section")
    # 2b. Compaction-proof copy: the typed focus Hub hands the governor renders into the
    # system-prompt anchor with the question and run A's disposition node.
    from orion.harness.task_anchor import build_focus_anchor

    anchor = build_focus_anchor(loop._focus_anchor(taken, RUN_A)) if taken is not None else None
    for needle in ("FOCUS ANCHOR", "How much of the bus traffic is heartbeats?",
                   f'MERGE (t:TurnOutcome {{run_id: "{RUN_A}"}})', '"set_aside: <why not>"'):
        if not anchor or needle not in anchor:
            failures.append(f"system-prompt focus anchor lacks {needle!r}")
    if asyncio.run(loop._take_focus(WorldviewSnapshot(), RUN_B)) is not None:
        failures.append("a second run took an already taken focus")

    # 3. Outcome.
    asyncio.run(loop._stamp_focus(RUN_A, turn_ok=True))
    row = next(r for r in table.rows if str(r["focus_id"]) == fid)
    if (row["status"], row["disposition"]) != ("ran", "set_aside:the bus logs for that hour were empty"):
        failures.append(f"stamped {row['status']}/{row['disposition']}")

    # 4. Read back through the curiosity tool.
    read_listener = CuriosityIntrospectListener(pool_provider=lambda: pool, reader_provider=lambda: None,
                                                source_ref=ServiceRef(name="orion-hub"), search=None)
    read_listener.bus = _Bus()
    seen = ask(read_listener, "curiosity", {"kind": "focus"})
    stories = {i.id: i.extra.get("story", "") for i in (seen.items if seen.ok else [])}
    story = stories.get(fid, "")
    if f"taken by run {RUN_A}" not in story or "set aside: the bus logs" not in story:
        failures.append(f"read tool story wrong: {story!r}")
    if "replaced by a newer focus" not in stories.get(hers.items[0].id, ""):
        failures.append("the replaced focus does not read as replaced")

    # 5. No disposition is unknown.
    ask(queue_listener, "queue_curiosity_focus", {"question": "What is the GPU idle floor?", "why": "fans"})
    asyncio.run(loop._take_focus(WorldviewSnapshot(), RUN_B))
    asyncio.run(loop._stamp_focus(RUN_B, turn_ok=True))
    seen_b = ask(read_listener, "curiosity", {"kind": "focus", "limit": 1})
    story_b = seen_b.items[0].extra.get("story", "") if seen_b.ok and seen_b.items else ""
    if "disposition: unknown" not in story_b or "explored" in story_b:
        failures.append(f"a run with no disposition did not read unknown: {story_b!r}")

    print(f"queued: {len(table.rows)} rows; Orion replacing Juniper's focus refused: {not refused.ok}")
    print(f"run A took: {fid}; second run took nothing: {'yes' if not any(f.startswith('a second') for f in failures) else 'no'}")
    print(f"story A: {story}")
    print(f"story B: {story_b}")
    for f in failures:
        print(f"FAIL {f}")
    print("PASS" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
