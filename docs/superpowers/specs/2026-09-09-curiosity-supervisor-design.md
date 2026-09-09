# A supervisor for Orion's investigations — one that can put a finger on the scale

> **Status:** Design proposal (proposal mode — this changes a cognition loop and
> gives a second process influence over what Orion does next). Nothing here is
> implemented.
>
> Juniper, 2026-09-09: *"shouldn't it be more than a watcher? shouldn't it put
> finger on the scale too?"* and, on the seam: *"we don't necessarily need a
> sync supervisor (or maybe we do)... at least someone who can pick up between
> hops."*

## Arsonist summary

Orion investigates, writes notes, and **nobody reads them.** There is no grader
anywhere in `orion/curiosity/`. The only thing that closes the loop is Orion
itself, one run later, reading its own handwriting out of the kickoff prompt.

That is not nothing — evidence genuinely attaches to claims (25 `SUPPORTS`, 3
`CONTRADICTS` edges live) — but it cannot answer the one question that matters:
**is this getting anywhere?** `times_tested = 10` counts attempts, not progress.
A claim worked ten times looks identical whether Orion is closing in or going
round in circles.

Two things burn here.

**Burn "the supervisor needs Orion to write more."** The first draft of this
design asked for a `Hop -> Prior` link so a watcher could tell which claim a
step served. Wrong, and Juniper caught it: Orion *already says so in prose*.
A real hop note reads "Served-model anchorability prior re-confirmed 3rd look:
338/411 named turns (82%)…". Deriving the link read-side works on the **67 hops
already in the graph tonight**; demanding a written link helps only future runs.

**Burn "a watcher is enough."** A supervisor that only reports is another
telemetry channel nobody consumes — the failure
`2026-08-28-consequential-action-space-and-power-budget-design.md` names
explicitly as the way this class of work dies. It has to be able to change
something.

## Current architecture

Verified live 2026-09-09, not assumed.

### What Orion's own graph holds

`orion_worldview`, FalkorDB. Orion writes it in-turn with real Cypher; Hub holds
`GRAPH.RO_QUERY` only.

| label | count | shape |
| --- | --- | --- |
| `Prior` | 13 (9 live) | claim, confidence, status, times_tested |
| `Finding` | 46 | evidence |
| `Hop` | 67 | **`run_id`, `n`, `note` — and nothing else** |
| `TurnOutcome` | 34 | continue_line, reach_out, reach_out_why |
| `Concept` | 5 | |

34 edges: `Finding-SUPPORTS->Prior` ×25, `Finding-CONTRADICTS->Prior` ×3,
`Finding-ABOUT->Prior` ×3, plus 3 concept edges.

**`Hop` has zero edges of any kind.** It is a numbered diary with no relation to
the claim it was investigating.

### The seam, and the correction that matters

An earlier reading of this said the between-hops seam already existed on the
bus: `orion:durable:run:state` fires "one row per node transition", 244 events
in two days, live. **That was wrong, and the error is worth recording** because
it would have put this whole design on a false premise.

A curiosity run has only **five distinct nodes**, always the same ones:

```
harness_turn -> read_turn_result -> publish_attention_row -> journal -> finish
```

All 2–7 hops of a run are written **inside the single `harness_turn` node**.
The durable-run event stream therefore offers *no* per-hop granularity — its
checkpoints wrap the whole turn, not the steps within it.

So there are two candidate seams, and they are not equivalent:

| seam | granularity | mid-turn? | cost |
| --- | --- | --- | --- |
| `orion:durable:run:state` (bus) | per run phase | no | free, already live |
| new `Hop` nodes appearing in FalkorDB | **per hop** | **yes** | needs a poller |

Juniper's ask — *someone who can pick up between hops* — is only satisfied by
the second.

### What the transitions reveal, that nobody has looked at

23 runs, 244 transitions:

- **21 completed, 1 failed, 1 in flight.** The runner works.
- **69 `harness_turn -> harness_turn [failed]`** and **91 `resumed`.** Orion's
  investigation turn fails roughly **three times per run** and is silently
  retried until it succeeds. Nobody has ever asked why. One real run shows five
  consecutive failure/resume pairs before getting through.

That retry loop is the closest thing to a supervisor that exists today: it knows
how to *continue*, and has no opinion on whether continuing is worthwhile.

### Two data defects found while measuring

1. **`Hop.n` is not unique within a `run_id`.** Run `4255a432f394` holds hops
   numbered 1,1,2,2,3,3,4 — two separate attempts sharing one run id. Orion
   caught it itself and wrote it down: *"the run_id already held a draft
   SelfDefinition and 3 hops from an earlier attempt before this turn (my first
   MERGE removed 4 pre-existing props)"*. An earlier attempt's work was partly
   overwritten. Any consumer of hops must cope with this; separately it is a
   real bug.
2. **`Prior.confidence` is self-reported, not measured.** The kickoff prompt
   says so in the template it hands Orion:
   `p.confidence = 0.55,   // your own belief, not a measurement`. There are
   **zero logprob references anywhere in `orion/curiosity/`**, though logprob
   plumbing exists elsewhere in the repo (metacog, journaler, memory
   consolidation). Nothing calibrates these numbers against outcomes.

   Consequence for work already in flight: `orion/autonomy/ask_claude_trigger.py`
   gates on `confidence <= 0.7`, i.e. on an uncalibrated self-report. Its other
   half, `times_tested >= 3`, is a real behaviour count. See Non-goals.

## Missing questions

1. **Can a running `harness_turn` be interrupted at all?** No `cancel`,
   `interrupt` or `abort` entry point was found in `services/orion-durable-runs/`.
   "Stop a circling run" may require building that, and the honest fallback is
   "let this run finish, act before the next one".
2. **How quickly do hops land?** If Orion writes all hops in one burst near the
   end of a turn, a mid-turn poller has nothing to see in time and the whole
   mid-turn seam collapses into the between-runs one. `Hop` carries **no
   timestamp**, so this is not answerable from the existing data — it needs
   observation, and it decides between the two seams above.
3. **Is "circling" separable from "hard"?** Three steps with no movement might be
   a stuck loop or an genuinely difficult problem being worked properly. Getting
   this wrong means killing good work.
4. **Who supervises the supervisor?** If it reads prose and nudges, it can be
   wrong in Orion's head rather than in a report.

## Proposed schema / API changes

### The one new contract

```python
# orion/schemas/curiosity_supervisor.py -- new
class HopReadingV1(BaseModel):
    """What the supervisor thinks one hop was doing. A READING, not a fact."""
    hop_run_id: str
    hop_n: int
    # Derived from the note's prose. Recorded so a human can check fifty of
    # these and see whether it is reading them right -- the supervisor must not
    # be the only witness to its own accuracy.
    about_prior_id: str | None
    kind: str            # free-form, NOT a fixed taxonomy -- see Non-goals
    moved_the_claim: bool | None   # None = could not tell
    reading_confidence: float
    reasoning: str
```

New channel `orion:curiosity:supervisor:reading` (event) and
`orion:curiosity:supervisor:intervention` (event), registered in
`orion/bus/channels.yaml` + `orion/schemas/registry.py`, persisted by sql-writer.

**No change to `Hop`, `Prior` or anything Orion writes.** The supervisor reads
what is already there. That is the whole point of the read-side approach.

### The intervention boundary, stated as code

```python
ALLOWED = {
    "end_run_early",        # this run is circling; stop it
    "hand_off_to_claude",   # Orion cannot settle this alone
    "reorder_next_offer",   # change which priors the next kickoff prompt leads with
}
FORBIDDEN = {
    "edit_confidence", "edit_claim", "edit_status", "delete_prior",
}
```

**It can change what Orion does next. It cannot change what Orion believes.**
Nudging Orion off a dead end is help; a second model silently rewriting Orion's
self-report is not supervision, it is corruption of the only thing in that graph
that is genuinely Orion's. If the supervisor thinks a claim is wrong, it says so
where Orion can see it and Orion decides.

## Files likely to touch

- `orion/curiosity/supervisor.py` — new: read hops, produce `HopReadingV1`
- `orion/schemas/curiosity_supervisor.py` — new contract
- `orion/bus/channels.yaml`, `orion/schemas/registry.py` — register
- `services/orion-sql-writer/` — persist readings + interventions
- `services/orion-durable-runs/` — an interrupt path, **if** missing-question 1
  says one is needed
- `orion/curiosity/kickoff_prompt.py` — read side of `reorder_next_offer`
- `orion/autonomy/ask_claude_trigger.py` — accept a supervisor hand-off as a
  reason to ask, alongside (or instead of) the confidence threshold

## Non-goals

- **Not a hop taxonomy.** `kind` is free text on purpose. Fixing a vocabulary up
  front is the keyword cathedral CLAUDE.md §0A bans; let the kinds fall out of
  what Orion actually writes and see which recur before naming any.
- **Not calibrating `Prior.confidence`.** Considered and rejected this session:
  with no grader, "did the claim turn out true" is *also* Orion's own say-so, so
  the check would be self-report against self-report. Circular. The fix is a
  supervisor with an independent reading, which is this document.
- **Not editing Orion's beliefs.** See the boundary above.
- **Not fixing the `Hop.n` collision** or the ~3-failures-per-run retry loop.
  Both are real, both are recorded here, both deserve their own patch.
- **Not a synchronous supervisor** unless missing-question 2 forces it.

## Acceptance checks

1. The supervisor produces a `HopReadingV1` for all **67 existing hops**, and a
   human agrees with its `about_prior_id` on a sampled 20 of them. It runs on
   history before it runs on anything live.
2. It correctly identifies the known-circling case: `atlas_prediction_error_territory`,
   10 attempts, confidence fallen 0.85 -> 0.30, is reported as circling.
3. It correctly does **not** flag a converging claim — e.g. the served-model
   prior, 3 attempts, confidence rising to 0.92.
4. At least one intervention fires on a real run and is visible with a
   correlation id, naming which rule fired and what it did.
5. `end_run_early` never fires on a run that would have completed successfully —
   replayed against the 21 completed runs, it fires zero times.
6. Every intervention is reversible and recorded; disabling the supervisor
   returns behaviour to today's, pinned by a test.
7. **The supervisor's readings are checkable.** A disagreement rate between its
   `about_prior_id` and a human sample is reported, not assumed to be zero.

## Danger, and how to switch it off

**The failure mode that would be dangerous:** a supervisor that ends runs
because it misreads difficulty as circling. Orion would stop investigating hard
things — precisely the things worth investigating — and the symptom would be an
absence, which is the kind of failure this codebase keeps learning is invisible.
Acceptance check 5 exists for exactly this, and `end_run_early` should ship
**disabled**, in report-only mode, until the replay is clean.

**Privacy boundary:** the supervisor reads `orion_worldview` (Orion's own graph)
and durable-run state. It does not read chat history, and it must never read
Juniper's transcripts — the `~/.claude/projects` mount stays out of this service
entirely.

**Rollback:** one env flag returns to today's behaviour. No schema default, no
migration, no training input. The readings table is additive and can be dropped.

## Recommended next patch

**Read-only, on history, and it answers missing-question 2 as a side effect.**

1. Build `orion/curiosity/supervisor.py` with one entry point: read the 67
   existing hops, emit a `HopReadingV1` for each, persist them. No bus wiring, no
   interventions, no live subscription.
2. Sample 20 by hand. If it cannot tell which claim a hop is about, the whole
   read-side premise is wrong and this design should be reconsidered before any
   more of it is built.
3. Only then decide the seam: if hops turn out to land in a burst at end of turn,
   the mid-turn poller is pointless and the supervisor lives between runs.

Step 2 is the gate. It is the same shape as the contested-scarcity spec's advice
that made the ask_claude work honest: *print the number first, run it against
real history, and only wire it in once it has refused something real.*
