#!/usr/bin/env python3
"""Orion's Day correction eval: does a recorded correction reach exactly one curiosity run, and
does the row afterwards say truthfully which named priors that run re-tested?

One correction is planted end to end over fixture data, through the real code at every step:

1. Record: Hub's record_correction responder (scripts/orion_day_correction_listener.py) over the
   fixture letter's ¶4 ("I said it dropped 4,059 requests"), naming two fixture priors. Two more
   requests must be refused with nothing written: an unknown prior id, a paragraph that does not
   exist. A third, naming no prior, is recorded as context only.
2. Offer: orion/orion_day/corrections.take_corrections for run A, rendered by the real
   build_kickoff_prompt: the section must show the ref, the claim, the correction, the evidence
   and each named prior's current confidence and times_tested, and the re-test instruction with
   run A's id. A second take (run B) must get nothing; a cancelled run C's take is released and
   offered again.
3. Run: a simulated outcome where run A re-tests (and moves) one named prior and never touches
   the other, scored by the real CuriosityInvestigation._stamp_corrections (turn-start snapshot
   from the spend log's own SQL, end state from the graph reader).
4. Check the row: the re-tested prior is in priors_retested and priors_moved; the untouched prior
   is absent from both; the context-only correction is stamped empty.
5. Next letter: gather_orion_day over the same table carries the corrections into material and
   the writer's digest cites [correction:<id>] with the outcome in plain words.

Deterministic and offline: no DB, LLM, bus, graph or network.

    python services/orion-hub/evals/run_orion_day_correction_eval.py

Exit 0 = every check passed, 1 = at least one failed.
"""
from __future__ import annotations

import asyncio
import json
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
RUN_C = "c3c3c3c3c3c3"


def _paths() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_HUB_ROOT)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_HUB_ROOT))


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- one linear scenario
    _paths()
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.curiosity.kickoff_prompt import build_kickoff_prompt, offered_priors_in_prompt
    from orion.curiosity.study_material import StudyMaterial
    from orion.curiosity.value import PriorState
    from orion.curiosity.worldview import WorldviewSnapshot
    from orion.introspect.transport import REQUEST_KIND, RESULT_PREFIX
    from orion.orion_day import corrections as oc
    from orion.orion_day import store
    from orion.orion_day.budget import build_llm_view
    from orion.orion_day.gather import gather_orion_day
    from orion.orion_day.tests import fixtures as fx
    from orion.schemas.introspect import IntrospectResultV1, IntrospectToolBindingV1
    from scripts.curiosity_investigation import CuriosityInvestigation
    from scripts.curiosity_offer_decisions import LOAD_TURN_SNAPSHOT_SQL
    from scripts.orion_day_correction_listener import OrionDayCorrectionListener

    letter = fx.reread_letter()
    before_states = {r["prior_id"]: PriorState.from_json(r) for r in fx.FIXTURE_PRIORS}

    class Conn(fx.CorrectionTable):
        async def fetchrow(self, sql, *args):
            if sql == store.SELECT_LETTER_SQL:
                return (letter.model_dump(mode="json") | {"letter_date": letter.letter_date}
                        if args[0] == letter.letter_date else None)
            if sql == LOAD_TURN_SNAPSHOT_SQL:
                return {"turn_snapshot": json.dumps([s.as_json() for s in before_states.values()])}
            return None  # the other gather sources: an empty day apart from the corrections

        async def fetch(self, sql, *args):
            from orion.orion_day import gather

            if sql in (oc.TAKE_CORRECTIONS_SQL, oc.OFFERED_FOR_RUN_SQL, gather.CORRECTIONS_SQL):
                return await super().fetch(sql, *args)
            return []  # every other gather source: an empty day apart from the corrections

    conn = Conn(now=datetime(2026, 9, 29, 16, 0, tzinfo=timezone.utc))
    pool = fx.TablePool(conn)
    reader = fx.FakePriorReader()
    failures: list[str] = []

    class Bus:
        def __init__(self):
            self.published = []

        async def publish(self, channel, envelope):
            self.published.append((channel, envelope))

    listener = OrionDayCorrectionListener(pool_provider=lambda: pool, reader_provider=lambda: reader,
                                          source_ref=ServiceRef(name="orion-hub"))
    listener.bus = Bus()
    binding = IntrospectToolBindingV1(invocation_context="unified_chat", parent_run_id="chat",
                                      parent_trace_id="chat-corr-eval", memory_allowed=True)

    def record(args) -> IntrospectResultV1:
        corr = uuid4()
        asyncio.run(listener.handle(BaseEnvelope(
            kind=REQUEST_KIND, correlation_id=corr, source=ServiceRef(name="orion-harness-governor"),
            reply_to=f"{RESULT_PREFIX}{corr}",
            payload={"operation": "record_correction", "binding": binding.model_dump(mode="json"), "args": args},
        )))
        return IntrospectResultV1.model_validate(listener.bus.published[-1][1].payload)

    # 1. Record.
    day = letter.letter_date.isoformat()
    base = {"ref": f"{day} ¶4", "claim": "I said it dropped 4,059 requests",
            "correction": "It dropped 12; 4,059 was the queue's lifetime total.",
            "evidence": ["Juniper said so in this conversation"],
            "prior_ids": [fx.CORRECTION_PRIOR, fx.UNTOUCHED_PRIOR]}
    planted = record(base)
    refused_prior = record({**base, "prior_ids": ["self:never_held"]})
    refused_ref = record({**base, "ref": f"{day} ¶40"})
    context = record({**base, "claim": "the service inventory held at 99", "ref": f"{day} ¶5",
                      "correction": "It held at 97.", "prior_ids": []})
    if not planted.ok:
        failures.append(f"planted correction refused: {planted.error}")
        print("\n".join(f"FAIL {f}" for f in failures))
        return 1
    if refused_prior.ok or refused_ref.ok:
        failures.append("an unknown prior or a missing paragraph was recorded")
    if len(conn.rows) != 2:
        failures.append(f"expected 2 rows (planted + context), got {len(conn.rows)}")
    cid = planted.items[0].id
    context_id = context.items[0].id if context.ok else None

    # 2. Offer.
    taken = asyncio.run(oc.take_corrections(pool, run_id=RUN_A))
    named = sorted({p for c in taken for p in c.prior_ids})
    current = oc.read_named_priors(reader, named)
    shown = tuple(c.__class__(**{**c.__dict__, "current_priors": current}) for c in taken)
    prompt = build_kickoff_prompt(StudyMaterial(generated_at=conn.now), view=WorldviewSnapshot(),
                                  run_id=RUN_A, corrections=shown)
    for needle in (oc.CORRECTIONS_HEADER, f"Letter {day} ¶4", '"I said it dropped 4,059 requests"',
                   "It dropped 12; 4,059 was the queue's lifetime total.", "Juniper said so in this conversation",
                   f'p.last_run_id = "{RUN_A}"', "A correction is evidence, not an override",
                   "No prior named: context only"):
        if needle not in prompt:
            failures.append(f"kickoff lacks {needle!r}")
    numbers = offered_priors_in_prompt(prompt)
    for row in fx.FIXTURE_PRIORS:
        want = (row["confidence"], row["times_tested"])
        if numbers.get(row["prior_id"]) != want:
            failures.append(f"kickoff shows {row['prior_id']} as {numbers.get(row['prior_id'])}, want {want}")
    if asyncio.run(oc.take_corrections(pool, run_id=RUN_B)):
        failures.append("a second run was offered an already offered correction")

    # A cancelled run's claim comes back.
    extra = record({**base, "correction": "Twelve, not 4,059.", "prior_ids": [fx.CORRECTION_PRIOR]})
    c_taken = asyncio.run(oc.take_corrections(pool, run_id=RUN_C))
    asyncio.run(oc.release_corrections(pool, run_id=RUN_C))
    again = asyncio.run(oc.take_corrections(pool, run_id=RUN_B))
    if [c.correction_id for c in c_taken] != [extra.items[0].id] or [c.correction_id for c in again] != [extra.items[0].id]:
        failures.append("a cancelled run's correction was not released and offered again")

    # 3. Run A re-tests (and moves) one named prior and never touches the other.
    after_rows = [dict(r) for r in fx.FIXTURE_PRIORS]
    after_rows[0].update(times_tested=after_rows[0]["times_tested"] + 1, confidence=0.3, status="revised",
                         last_run_id=RUN_A)
    loop = CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=14400.0, daily_cap=3, timeout_sec=60.0,
        session_id="eval", crystallization_sample=1, relation_sample=1, pool_provider=lambda: pool,
        source_ref=ServiceRef(name="orion-hub"), reader=fx.FakePriorReader(after_rows),
        orion_day_corrections_enabled=True,
    )
    asyncio.run(loop._stamp_corrections(RUN_A))

    # 4. The rows.
    by_id = {str(r["correction_id"]): r for r in conn.rows}
    row = by_id[cid]
    retested, moved = row["priors_retested"], row["priors_moved"]
    if retested != [fx.CORRECTION_PRIOR]:
        failures.append(f"priors_retested={retested}, want [{fx.CORRECTION_PRIOR}]")
    if fx.UNTOUCHED_PRIOR in (retested or []) or fx.UNTOUCHED_PRIOR in (moved or []):
        failures.append("the untouched named prior was recorded as re-tested or moved")
    if moved != [fx.CORRECTION_PRIOR]:
        failures.append(f"priors_moved={moved}, want [{fx.CORRECTION_PRIOR}]")
    if row["outcome_stamped_at"] is None:
        failures.append("outcome_stamped_at not set")
    if context_id and (by_id[context_id]["priors_retested"], by_id[context_id]["priors_moved"]) != ([], []):
        failures.append("the context-only correction was not stamped empty")

    # 5. The next letter.
    material = asyncio.run(gather_orion_day(conn, letter.letter_date, now=datetime(2026, 9, 30, 15, tzinfo=timezone.utc)))
    view = build_llm_view(material)
    if f"correction:{cid}" not in view.included_refs:
        failures.append("the next letter's digest does not cite the correction")
    if f"not re-tested: {fx.UNTOUCHED_PRIOR}" not in view.digest_md:
        failures.append("the next letter's digest hides the untouched prior")

    print(f"recorded: {sum(1 for r in conn.rows)} rows; refused without writing: "
          f"{int(not refused_prior.ok) + int(not refused_ref.ok)}/2")
    print(f"offered to run A: {len(taken)}; offered again to run B: 0 expected; released and re-offered: "
          f"{'yes' if again else 'no'}")
    print(f"stamped: retested={retested} moved={moved} (untouched named prior: {fx.UNTOUCHED_PRIOR})")
    print(f"next letter material corrections: {len(material.corrections)}")
    for f in failures:
        print(f"FAIL {f}")
    print("PASS" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
