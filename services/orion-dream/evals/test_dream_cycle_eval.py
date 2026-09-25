"""Dream cycle v2 behavior eval: rest point, honesty under a refusing LLM, blind arms.

These are the properties the metric gate (CLAUDE.md §0A) asks of a new signal
and a new cognition-shaped output:

  rest point   sleep pressure returns to exactly 0.0 right after a cycle and
               only rises as new unprocessed rows arrive -- not a permanent
               floor, not a decay artifact.
  honesty      an LLM that finds no links yields zero hypotheses and a counted
               no_link, never filler.
  blind arms   dream and control pairs go through the identical prompt, so the
               only difference the scorecard can see is how pairs were chosen.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

T0 = datetime.now(timezone.utc) - timedelta(hours=12)


def _corpus():
    """Rows with timestamps; the fake loader applies `since` like the SQL does."""
    rows = []
    for i in range(6):
        rows.append(("metacog", T0 + timedelta(minutes=10 * i), {
            "id": f"m{i}", "summary": f"self-observation {i}: recall miss on topic {i}",
            "severity": "critical" if i % 2 else "degraded", "trigger_kind": "recall", "tags": [f"t{i}"],
        }))
    for i in range(4):
        rows.append(("crystallization", T0 + timedelta(minutes=5 * i), {
            "crystallization_id": f"c{i}", "subject": f"concept {i}", "summary": f"summary {i}",
            "salience": 0.5 + 0.1 * i, "tags": [f"k{i}"],
        }))
    rows.append(("resonance", T0, {"alert_id": "a0", "theme_key": "being watched", "violation_count": 3}))
    rows.append(("compaction_request", T0, {"request_id": "r0", "theme": "sleep", "reason": "recurs"}))
    return rows


class _World:
    def __init__(self, answer):
        self.rows = _corpus()
        self.last_start = None
        self.last_end = None
        self.answer = answer
        self.prompts = []

    def deps(self):
        from app.cycle import CycleDeps

        def load(since, limit):
            out = {"metacog": [], "compaction_request": [], "resonance": [], "crystallization": []}
            for kind, ts, row in self.rows:
                if ts > since:
                    out[kind].append(row)
            return out

        def persist(cycle):
            self.last_end = cycle.ended_at
            if cycle.status != "failed":
                self.last_start = cycle.started_at
            return True

        async def complete(prompt):
            self.prompts.append(prompt)
            return self.answer(prompt)

        return CycleDeps(
            load_source_rows=load,
            load_idle_minutes=lambda: 600.0,
            load_last_window_start=lambda: self.last_start,
            load_last_attempt_end=lambda: self.last_end,
            persist_cycle=persist,
            complete=complete,
        )


def _linker(_prompt):
    return json.dumps({"link": True, "claim": "Recall misses cluster around the same concepts each night", "why": "w"})


def test_pressure_rest_point_is_exactly_zero_after_a_cycle_and_rises_with_new_rows():
    from app.cycle import read_pressure, run_cycle_once

    world = _World(_linker)
    before, _ = read_pressure(world.deps(), datetime.now(timezone.utc), world.last_start)
    assert before.pressure > before.threshold

    cycle = asyncio.run(run_cycle_once(world.deps()))
    assert cycle is not None and cycle.status == "completed"

    after, _ = read_pressure(world.deps(), datetime.now(timezone.utc), world.last_start)
    assert after.pressure == 0.0 and after.counts == {}

    world.rows.append(("metacog", datetime.now(timezone.utc) + timedelta(seconds=1), {
        "id": "late", "summary": "new surprise", "severity": "critical", "trigger_kind": "x", "tags": [],
    }))
    later, _ = read_pressure(world.deps(), datetime.now(timezone.utc), world.last_start)
    assert later.pressure == 1.0


def test_gateway_outage_keeps_the_backlog():
    from app.cycle import read_pressure, run_cycle_once

    def down(_p):
        raise RuntimeError("gateway down")

    world = _World(down)
    cycle = asyncio.run(run_cycle_once(world.deps()))
    assert cycle.status == "failed"
    still, _ = read_pressure(world.deps(), datetime.now(timezone.utc), world.last_start)
    assert still.pressure > still.threshold  # nothing was thrown away


def test_refusing_llm_yields_no_hypotheses_and_counts_no_link():
    from app.cycle import run_cycle_once

    world = _World(lambda _p: json.dumps({"link": False}))
    cycle = asyncio.run(run_cycle_once(world.deps()))
    assert cycle.hypotheses == []
    assert cycle.no_link_count == len(world.prompts) > 0
    assert cycle.llm_failures == 0


def test_arms_share_one_prompt_template_and_control_count_is_exact():
    from app.cycle import run_cycle_once
    from app.recombine import PROMPT
    from app.settings import settings

    world = _World(_linker)
    cycle = asyncio.run(run_cycle_once(world.deps()))
    head = PROMPT.split("{kind_a}")[0]
    assert all(p.startswith(head) for p in world.prompts)
    arms = [h.arm for h in cycle.hypotheses]
    assert arms.count("control") == settings.DREAM_CONTROL_PER_CYCLE
    assert arms.count("dream") == settings.DREAM_HYPOTHESES_PER_CYCLE
    # every hypothesis is traceable to two real replay/candidate refs
    refs = {r for _, _, row in world.rows for r in row.values() if isinstance(r, str)}
    for h in cycle.hypotheses:
        assert h.ref_a.split(":", 1)[1] in refs and h.ref_b.split(":", 1)[1] in refs
