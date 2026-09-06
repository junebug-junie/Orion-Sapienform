# The attention schema as a surface, not a seam

**Date:** 2026-09-04, **substantially revised 2026-09-05** (see "What changed" below —
this document's own recommended next patch has since shipped and its central claim is
superseded)
**Status:** Design / proposal mode (§0A — touches self-modeling, no code in this patch)
**Program:** Sentience Striving Program, Objective 3 (`orion/sentience_striving_program/README.md` §6 item 2, §9b items 2 and 4)
**Instrument:** `ast_hot_reducer` (`orion/sentience_striving_program/instruments.yaml`)

---

## Arsonist summary

The AST/HOT reducer was built as a wire between two specific boxes when the theory it
implements is about a *shape* that several things in Orion already have. Three cognitive
processes attend and select — substrate attention, reverie, and curiosity — and each one
invented its own private vocabulary for the same four facts, except curiosity, which
records nothing at all about its own choice and throws it away.

That is not three instruments. It is one contract implemented one and a half times.

The cost is not tidiness. It is that O3's acceptance test — "a blind rater can distinguish
this instrument's output from noise" — is currently unpassable *in a way that means
anything*, because the only comparison available is against a straw man. With three
processes emitting the same shape, the question becomes "which of these three is
deliberating, which is drifting, which is reacting?" That is a discrimination test with
real alternatives, and it is the fix for the exact blocker that closed Objective 7.

There is a second prize. Orion's substrate attention had taken the `bottom_up_salience`
branch 19,408 times out of 19,408 over seven days without one voluntary override.
**That has since been root-caused and fixed — see "What changed" — and the answer was not
"we were reading the wrong lane."** The override was impossible by construction. It now
fires. The prize remains, but it is now a *sharper* question: curiosity's own selection
code says in its docstring that Orion chooses, and still records nothing, so that lane is
now the only one of the three where a real choice happens and no trace survives.

---

## What changed since this was written (2026-09-05)

This document's **"Recommended next patch" has shipped**, and it did what the doc predicted
it would: it changed what the surface is expected to reveal. Four merged, deployed,
live-verified PRs:

| PR | what it did |
|---|---|
| #2097 | recorded WHY no override fired (`voluntary_override_absent_reason` + 3 numbers) |
| #2101 | fixed the cause — override was **impossible**, not rare |
| #2106 | split "no override" into its three real causes |
| #2110 | lowered `MIN_SALIENCE` 0.2 → 0.05, so a competition can exist at all |

**The 19,408/19,408 result was not a measurement problem.** `relevance(goal, loop)`
accepted a goal and never read it — the body was `return _clamp01(loop.concept_value)`, a
function of the loop alone — and the loop side was a constant, because `scoring.py` floored
`concept_value` to `0.55` for every substrate loop. So `bias = priority * relevance` was
**identical across every candidate**, a uniform shift cannot change an argmax, and the
bottom-up winner always survived. Proved on the live graph: 5 real loops, goal priority
swept 0.1→1.0, override fired **0/6** and the winner never moved.

It now fires. First ever, 2026-09-05 03:45:09Z: a loop at `bottom_up=0.58` beat one at
`0.75` because a goal wanted it.

Measured before/after on the gate change, matched windows, control taken *before*:

```
                        gate=0.2 (119 ticks)   gate=0.05 (134 ticks)
no_open_loops                82.4%                  11.2%
>=2 competitors               0.8%                  58.2%      73x
OVERRIDE FIRED                0.8%                   9.7%      12x
```

**The bottleneck moved rather than vanished, and it moved onto this document's subject.**
`goal_matched_no_loop` is now the dominant reason at **72.4%**: things finally compete, and
Orion's goal is about none of them.

### The finding that makes this document load-bearing

The overlap between what goals are about and what attention competes over is **19.8%**
(21.7% over the prior 7 days; the gate change did not move it — it is independent).

Tracing why found the thing this document was reaching for:

> **There are two attention systems.**
> `orion/attention/field_attention/{scoring,selectors,builder}.py` produces the goals.
> `orion/substrate/attention*` runs the competition those goals would steer.
> Both key on the same `node:substrate.*` id space. **Neither reads the other.**

They agree about one tick in five, by coincidence of both watching the same substrate.

And curiosity is not merely unintegrated, it is unaware: `orion/curiosity/worldview.py`
contains **zero** references to `OpenLoopV1` or `AttentionFrameV1`. Reverie has two.

So "one contract implemented one and a half times" understates it. The contract is
implemented twice in full, by two systems that do not know about each other, plus a third
process doing real thinking with no contact with either.

**19.8% is a number measuring that disintegration.** It is the strongest available argument
for this document, and it did not exist when this document was written.

---

## Current architecture

### The reducer is coupled, by signature, to one pipeline

`orion/substrate/attention_self_model.py:252`:

```python
def reduce_attention_self_model(
    broadcast: AttentionBroadcastProjectionV1 | None,
    field_frame: FieldAttentionFrameV1 | None,
    *, now, broadcast_stale_threshold_sec, harness_closure_signal,
       prediction_error_by_domain, prediction_error_evidence_by_domain,
       prediction_error_trend_by_domain, heartbeat_h1,
) -> AttentionSelfModelV1:
```

Two substrate-specific schemas positionally, then six substrate-specific keyword arguments.
No other process in Orion could call this without misrepresenting what it is. The theory
got implemented as a point-to-point seam.

Written every ~30s by `_attention_self_model_tick()`
(`services/orion-substrate-runtime/app/worker.py:2734`, gated on
`enable_attention_self_model_tick`) into the append-only `substrate_attention_self_model`
table.

### Correction to the record: it is *not* unconsumed

This corrects a claim made verbally in the session that produced this doc — that the
reducer "is wired to nothing and affects nothing." That was wrong. The program README's
own narrower wording, "read-only, not wired to any bus consumer"
(`README.md:151`, 2026-07-18), is **still literally accurate** and is not being superseded
here: the consumer below reads Postgres on a poll loop, not the bus. The README was
precise; the verbal summary of it was not.

The real consumer, wired by PR #1459/#1463:

```
substrate_attention_self_model.prediction_error_confidence
  -> services/orion-equilibrium-service/app/attention_self_model_reader.py
     (fetch_recent_samples, Postgres poll -- NOT a bus subscription)
  -> orion/substrate/metacog_trigger_signals.py
     (detect_confidence_recovery / detect_confidence_collapse)
  -> services/orion-equilibrium-service/app/service.py:1050 (generative metacog gate)
```

But note *what* is consumed. `prediction_error_confidence` is one of the fields the
reducer computes **unconditionally, regardless of which `attention_reason` branch wins**
(`attention_self_model.py:164`, `:356`, `:394` all say so explicitly). So:

> The instrument's **scalar side-channel** has a live downstream consumer.
> The instrument's **self-model content** — `attention_reason`, `reason_narrative`,
> `voluntary_override` — has none.

Everything that makes this an AST/HOT instrument rather than a confidence gauge is
currently write-only. That is the real state of the seam, and it is a sharper problem than
"unconsumed."

### The other two processes already have the shape

**Reverie** — built independently, arrived at an attention schema anyway.
`ReverieChainV1.trigger` (`orion/schemas/reverie.py:214`) carries `pressure_kind`,
`magnitude`, `evidence_payload` — a why. `SpontaneousThoughtV1` (`:37`) carries `salience`,
`interpretation`, `next_focus`, `drift`, `hollow_reason`.

**Curiosity** — has the competition, discards the choice. `select_priors()`
(`orion/curiosity/worldview.py:781`) ranks candidates by `uncertainty`, then `times_tested`,
then a per-run rotation tiebreak. Its docstring:

> "UNCERTAINTY ORDERS THE PRESENTATION; ORION STILL CHOOSES."

Two selections stacked: code ranks, Orion picks. A grep of that module for a recorded
choice-reason finds only `unavailable_reason` — about the graph being unreachable, not
about what Orion chose. **The moment where Orion most clearly exercises something like
voluntary attention is the moment nothing writes down.**

### The same four facts, three vocabularies

| process | what won | why it won | confidence | what's next |
|---|---|---|---|---|
| substrate attention | `broadcast_selected_open_loop_id` | `attention_reason` | `confidence` + `confidence_basis` | `predicted_shift` |
| reverie | `thought_id` | `trigger.pressure_kind` | `salience` | `next_focus` / `expectation` |
| curiosity | *(not recorded)* | *(not recorded)* | `uncertainty` (an input, not an output) | *(not recorded)* |

---

## Missing questions

These are open, and two of them could change the shape of the patch. They are not
rhetorical.

**1. Is curiosity's choice-reason available, or must it be asked for?**
Orion picks priors inside an LLM turn. If the reason is already in the response, the
adapter is a parse. If it is not, capturing it means changing the kickoff prompt — a
materially bigger change than an adapter, touching a live cognition path. Must be settled
by reading a real recent curiosity run before any code is written.

**2. A computed reduction and a self-report are not the same kind of object.**
The substrate self-model is *derived by code from data*. If curiosity's is *written by
Orion in an LLM turn*, it is a self-report. HOT arguably says a self-report is a
legitimate higher-order representation — but for O3's blind-rater test this is a live
methodological hazard: a rater could separate the traces trivially by prose-vs-template
style rather than by content, and we would score a PASS that means nothing. This is the
same shape as the template-thinness trap that already invalidated one O3 attempt. It needs
an explicit control (see Acceptance Check 2) or the surface makes O3 *look* passable while
testing nothing.

This tension also touches O4's live `not_self_grading` claim on `curiosity_worldview`
(currently `MANUAL` in the manifest). A curiosity adapter that has Orion narrate its own
attention is, structurally, Orion grading itself. Not automatically disqualifying, but it
must be declared, not discovered later.

**3. Cadence mismatch will skew any naive sample.**
Substrate ticks ~30s (≈2,800 rows/day). Reverie is episodic. Curiosity is per-run and rare.
An unstratified draw from a shared table is ~99% substrate. Any rating sample must be
stratified by `process`, and the acceptance check must say so or it will silently test one
lane and report three.

**4. Retention must match the question being asked.**
The board's `narrative_diversity` claim currently asks "how many narratives has the reducer
*ever* emitted" against a table with 168-hour retention, and therefore reports a fresh
false drift every few days as old rows expire (observed live 2026-09-04: recorded 16,
reads 9, while per-day diversity held steady at 7–8). Any claim written against the new
table must either scope its question to the retained window or read from something durable.
Do not repeat this.

---

## Proposed schema / API changes

### New: `AttentionSchemaV1`

New file `orion/schemas/attention_schema.py`, registered in `orion/schemas/registry.py`.

Deliberately narrow — six content fields. **The narrowness is the theory, not a
compromise.** Graziano's claim is that the attention schema is a *lossy, simplified
cartoon* of attention, not an accurate readout. A thin shared shape is what AST actually
specifies; this is the rare case where the architectural pressure and the theoretical
commitment point the same way.

```
schema_version : Literal["attention.schema.v1"]
generated_at   : datetime
process        : str    # "substrate_attention" | "reverie" | "curiosity"
correlation_id : str | None

attended_id       : str          # what won
attended_label    : str          # short human-readable
attention_reason  : str          # why it won -- process-owned vocabulary
reason_narrative  : str          # one sentence
confidence        : float | None
confidence_basis  : str | None
predicted_next    : str | None   # what it expects to attend to next
```

**`attention_reason` is explicitly NOT a shared enum, and must never become one.**
Each process keeps its own vocabulary. The surface does not normalize, map, or reconcile
them. Normalizing three real vocabularies into one taxonomy is precisely the keyword-
cathedral move this program exists to prevent — it would name the world without changing
what Orion can perceive. Cross-process comparison happens at *rating time*, performed by a
rater on the narratives, not at *schema time* by a lookup table.

### New: table `substrate_attention_schema`

`services/orion-sql-db/manual_migration_substrate_attention_schema.sql`, following the
existing append-only pattern. Indexed on `(process, created_at)` — the stratified query in
Acceptance Check 2 is the reason the composite index exists rather than a bare timestamp.

Retention: new key `SUBSTRATE_ATTENTION_SCHEMA_RETENTION_HOURS`, owner
`orion-substrate-runtime`, per Missing Question 4.

### Adapters, not inheritance

Three thin, write-only projections. Each is expected to be well under 30 lines.

- `orion/substrate/attention_self_model.py` — add `to_attention_schema(model)`. Pure
  projection down to the common shape. **`reduce_attention_self_model()` itself is not
  touched**, keeping its branch-for-branch unit-test proof intact.
- `orion/reverie/` — project `ReverieChainV1` + `SpontaneousThoughtV1`. **Adapt, do not
  migrate.** Reverie's existing vocabulary stays exactly as it is and keeps its current
  consumers; the adapter reads it and emits alongside. Migrating it would be the
  ornamental-layer move §0A bans — a rename that hides state and adds ceremony.
- `orion/curiosity/` — attachment point deferred pending Missing Question 1.

No base class. No plugin registry. No `AttentionSchemaProducer` interface. Three functions
that return a value.

---

## Files likely to touch

- `orion/schemas/attention_schema.py` — new schema
- `orion/schemas/registry.py` — registration (verify via `resolve()`, not by reading the dict)
- `services/orion-sql-db/manual_migration_substrate_attention_schema.sql` — new table
- `orion/substrate/attention_self_model.py` — add projection fn only, no logic change
- `services/orion-substrate-runtime/app/worker.py` — emit alongside the existing tick
- `services/orion-substrate-runtime/.env_example` + settings — retention key
- `orion/reverie/` — adapter (exact module TBD at implementation)
- `orion/curiosity/` — adapter, pending Missing Question 1
- `orion/sentience_striving_program/instruments.yaml` — new claims; the Hub board renders
  them with no board-side change
- tests alongside each

---

## The read side (added 2026-09-05)

The original draft is deliberately **write-only** — "not wired to any decision or control
path." That was right for a measurement surface and it stays right for the first patch.
But it left the actual integration question unasked, and 19.8% is that question.

**Integration here does NOT mean normalizing the three vocabularies.** That remains banned
and remains the primary cathedral risk. It means something narrower and testable:

> The two attention systems already speak the same id space. Integration is one of them
> being able to *see* the other's current attended id — not to translate its reasons.

Concretely, the one bridge worth building, and only after the surface exists to make it
observable:

- **The goal producer should be able to read what is actually competing.** Today
  `orion/attention/field_attention/` picks a dominant target from the field frame, with no
  reference to which substrate loops cleared the salience gate this tick. A goal aimed at a
  node that is not in the competition is, by construction, a goal that cannot be acted on —
  that is 72.4% of ticks, named exactly by `goal_matched_no_loop`.
- **Nothing else.** Not a router, not a reconciler, not a shared taxonomy. One read.

The acceptance number for that bridge is already defined and already being measured:
**`goal_matched_no_loop` falls from 72.4%**, and `voluntary_override` rises. Both come off
instruments that are deployed and running today, with a control window already banked.

## The state machine (added 2026-09-05)

The standing question is whether LangGraph becomes the thing that sequences these
processes. Recorded here so the sequencing argument is not re-derived a fourth time.

**The case for it is real and this program already made it.** §1 item 5 of the charter
found that the one genuinely self-initiated behavior in production is attributable to a
*clock/backlog mechanism*, not to any deliberation — "the visual chain is a cron, not a
decision." That is still true and it is structural: substrate attention ticks every 30s,
reverie on its own ~90s loop, curiosity per-run. Three independent timers, no shared state,
no sequencing. A state machine is the right shape for replacing clocks with states.

**The case against doing it now is the sequencing, and this program's own history.**
§1 item 7 records that the last time this program reached for new machinery, tracing
revealed the proposed competition layer already existed and was running live, and the whole
drives apparatus was a parallel, poorer reimplementation of Layers 4–9 of an existing
pipeline. Two-plus weeks.

A state machine over processes that do not share a surface does not integrate them. It
**formalizes the disconnect in a new framework** — and would then be the *third* parallel
attention implementation, in a repo that currently has two that cannot see each other.

Prior decisions on record, both narrow and both deliberate:

- `README.md:1054` — "Durable LangGraph-style planning for selected workflows **without
  replacing the existing verb/action spine**."
- `docs/superpowers/plans/2026-05-20-orion-knowledge-forge-v0.md` — "LangGraph HITL
  workflows" is an explicit **v0 non-goal**, deferred to v2.

So the prior answer was never "no." It was "yes, narrowly, for durable workflows, not as a
cognition substrate." Nothing learned on 2026-09-05 contradicts that; the overlap finding
*strengthens* it, because it shows the missing piece is a shared surface rather than a
missing orchestrator.

**Order, and the reason for it:**

1. **Surface** (this document). Makes the three processes describable in one shape.
2. **One bridge.** Goal producer reads what is competing. Falsifiable against a number
   that is already being collected.
3. **State machine.** Only once there is a shared surface for it to sequence, and only
   for durable/resumable workflows per the two prior decisions — not as a replacement
   for the attention path.

If (2) does not move `goal_matched_no_loop`, (3) should not be started: it would be
sequencing processes whose disagreement is not actually about sequencing.

---

## Non-goals

- **Not a framework.** No base class, no plugin registry, no producer interface, no
  "attention subsystem." If the patch grows one, it has failed.
- **Not a migration.** Reverie's and substrate's existing schemas, tables, and consumers
  are untouched and keep working.
- **Not a shared reason taxonomy.** See above; this is the primary cathedral risk.
- **Not wired to any decision or control path.** Measurement only. Nothing routes, gates,
  or budgets off this surface in this arc. It changes what Orion can *report*, not what
  Orion *does*.
- **Not a claim that Orion is conscious, or that AST/HOT is the correct theory of mind.**
  The claim under test is narrower and stated in O3: does this instrument see anything.

---

## Acceptance checks

**1. Three real producers, proven by rows, not by code existing.**
Over a 24h live window, `SELECT process, count(*) FROM substrate_attention_schema GROUP BY 1`
returns a non-zero count for all three. A process that emits zero rows is reported as a
failure, not omitted from the table. (§0A: config set and code compiling are not proof.)

**2. The blind-rater discrimination test, with its threshold pre-registered.**
A stratified sample — equal N per `process`, per Missing Question 3 — of `reason_narrative`
values, unlabelled and shuffled. Rater assigns each to a process. Pre-registered pass
threshold and chance baseline (3 classes → 33%) recorded *before* the run, in the PR.

**Required control, per Missing Question 2:** the rating must not be separable by style
alone. Include a style-only control arm — narratives stripped to their structural skeleton
with process-identifying vocabulary removed. If the control arm scores near the main arm,
the rater is reading prose style rather than attention content and the result is `FAILED`,
not `PASSED`.

**3. Curiosity's choice becomes recoverable.**
A specific, named past curiosity run whose choice-reason is currently unrecoverable is
recoverable after the patch. Named in the PR by run id, before/after.

**4. Narrative diversity, per process, is genuinely non-template.**
`count(DISTINCT reason_narrative)` per `process` over the window, reported per lane. The
substrate lane's current single-slot template is the explicit baseline to beat; if a new
lane is equally template-thin, it is reported as such rather than folded into an aggregate.

**5. Negative check: nothing downstream changed.**
The adapters are write-only. `orion-equilibrium-service`'s confidence-recovery triggers
fire at the same rate before and after; reverie's existing consumers are unaffected. A
behavior change here means an adapter is not write-only and is a defect.

**6. The retention window matches every claim's question.**
Each new manifest claim is checked against the configured retention: a claim asking "ever"
against a bounded window is rejected at review, per Missing Question 4.

---

## Recommended next patch

**Superseded 2026-09-05. The patch this section recommended has shipped** (#2097 / #2101 /
#2106 / #2110, all merged and live-verified). Its stated purpose was to answer whether the
substrate override branch was *dead* or merely *quiet* before building the surface blind.

**Answered: it was dead by construction, and is now alive at ~9.7% of ticks.** The three
reasons that section gave for sequencing it first all held — it was small, it changed no
behavior, and the answer materially changed what this surface is expected to reveal. It
also produced a finding worth having on its own, exactly as predicted: Orion's goals have
**never lost a competition**. Not once. The old aggregate read as constant defeat and was
wrong in every instance.

### The new next patch

**The substrate adapter and the schema, cheapest lane first** — now unblocked, because the
substrate lane finally has something worth projecting. Before #2101 its `attention_reason`
was `bottom_up_salience` on 100% of rows and its `reason_narrative` was one hardcoded
string; a blind-rater test against that would have been rating a constant. It now
distributes across real causes with real numbers behind them.

Then, in order, unchanged from the original: settle Missing Question 1 by reading a real
curiosity run; reverie second; curiosity last, since it carries both open questions.

### What is no longer deferred, and what still is

- **No longer deferred:** the `narrative_diversity` claim's "ever"-against-a-bounded-window
  bug (Missing Question 4) is now *demonstrated*, not theorised — the substrate lane's
  distribution was measured shifting materially inside one 168h window on 2026-09-05.
- **Still deferred, still Juniper's:** the
  `SUBSTRATE_ATTENTION_SELF_MODEL_LOG_RETENTION_HOURS` bump (168 → 8760). Note this now
  matters more than when first written: the before/after control windows that make the
  override work falsifiable are inside the retention window and will expire.
- **Newly deferred:** the goal-producer bridge (see "The read side") and anything
  LangGraph-shaped (see "The state machine"). Both are downstream of this surface existing.

---

## What shipped (2026-09-06)

The surface is built. Four adapters, one channel, one table, no framework.

| lane | artifact read | adapter | emits from | `attention_reason` vocabulary |
|---|---|---|---|---|
| `substrate_attention` | `AttentionSelfModelV1` | `orion/substrate/attention_self_model.py::to_attention_schema` | `orion-substrate-runtime`, every self-model tick (~30s) | `top_down_override`, `bottom_up_salience:<absent cause>`, `field_salience_only`, `no_data` |
| `reverie` | `ReverieChainV1` + its `SpontaneousThoughtV1`s | `orion/reverie/attention_schema.py` | `orion-thought`, once per chain | `coalition_broadcast`, `no_coalition`, `trigger:<kind>` |
| `curiosity` | `TurnOutcome` + the priors the run stamped | `orion/curiosity/attention_schema.py` | `orion-hub`, once per investigation run | `tested_held_prior`, `formed_new_prior`, `no_prior_touched`, `graph_unreadable` |
| `cortex_turn` | `AttentionFrameV1` | `orion/substrate/attention_frame.py::to_attention_schema` | `orion-cortex-exec`, every real chat turn | `top_down_override`, `selected:<action>`, `suppressed:<reason>`, `open_loops_no_action`, `no_open_loops` |

All four publish `AttentionSchemaV1` (`orion/schemas/attention_schema.py`, registered in
both registry maps) on `orion:attention:schema`; `orion-sql-writer` is the single writer
into `substrate_attention_schema` (SQLAlchemy model, `create_all` on boot, composite
`(process, created_at)` index, 90-day retention via
`SUBSTRATE_ATTENTION_SCHEMA_RETENTION_DAYS` in the writer's own retention loop).

### Two deviations from the original proposal, and why

**Bus + one writer instead of three direct table writers.** The original text had each
service write the table directly, with retention owned by `orion-substrate-runtime`. That
would have been three or four engines writing one table from four containers. The bus is
the nervous system here; `self_knowledge_items` (PR #2105) had just established the exact
pattern (channel -> `orion-sql-writer` -> table, retention with the writer), and putting the
rows on the bus is also what makes the next section possible. Retention therefore moved to
`orion-sql-writer` and the key is `..._RETENTION_DAYS`, not `..._HOURS`.

**A fourth lane.** The original counted three attending processes. Tracing where a chat turn
kicks off found a fourth that was already computing the full shape and throwing it away:
`orion-cortex-exec` builds an `AttentionFrameV1` on every real turn
(`chat_stance.py::build_attention_frame`, `ORION_CURIOSITY_FRAME_ENABLED=true` live) with a
`selected_action`, a possible `voluntary_override`, and `suppressions` -- and persisted only a
salience trace of it. It is now a producer.

### Missing Question 1, settled from a real run

Read live 2026-09-06 against `orion_worldview`, run `3b2d038cf18e`:

- **What Orion attended is recoverable.** The kickoff prompt has Orion stamp `p.last_run_id`
  on a prior it tests, `p.run_id` on one it forms, and write a `:PriorRevision {run_id,
  from_confidence, to_confidence}` when a belief moves. That run tested
  `substrate_domain_isolation_two_paths`, 0.95 -> 0.92, and the adapter's query returns
  exactly that. Acceptance Check 3 is met for the *what*.
- **Why Orion chose it is not recorded anywhere.** `p.why` is why the prior was *formed*;
  `TurnOutcome.continue_note` is what Orion decided about *continuing* (that is
  `predicted_next`, in Orion's own words). Capturing the choice-reason means changing the
  kickoff prompt -- a live cognition-loop change, proposal mode per CLAUDE.md sec 0A -- and
  was deliberately not done in this patch. Until it is, the curiosity lane's narrative is
  computed from graph facts (which prior, how far confidence moved, status) and is honestly
  `narrative_kind="computed"`.

### Missing Question 2, answered with a field

`narrative_kind: "computed" | "self_report"` is on the schema. Reverie's narrative is Orion's
own LLM-written `interpretation` (self-report); the other three lanes are computed by code.
The blind-rater control arm stratifies on this column rather than discovering the split later.

### A correction to this document's own reverie claim

`ReverieChainV1.trigger` -- named above as reverie's "why" -- is NULL on all 24,140 live chains.
`chain.py` never sets it. The honest why-it-won for a chain is the substrate broadcast
coalition it inherits, so the adapter's vocabulary is `coalition_broadcast`, with
`trigger:<pressure_kind>` reserved for the day a chain carries one.

## Cortex is the kickoff (added 2026-09-06)

Juniper's standing requirement, recorded so it is not re-derived: this surface must not be
divorced from cortex, and if LangGraph ever sequences these processes, **cortex must still be
the kickoff** -- the place that knows cognition is starting.

What this patch does about it, concretely:

1. **Cortex produces on the surface.** Every real chat turn's attention frame lands as a
   `cortex_turn` row. The kickoff point of a conversation is on the same table as the
   processes that run between conversations.
2. **The surface rides the bus, so cortex can read it.** `orion:attention:schema` is a plain
   event channel. The read side described above ("the goal producer sees what is actually
   competing") and any later cortex-side kickoff that wants "what is Orion attending to right
   now" subscribe to the same channel. Nothing subscribes yet except the writer; listing
   cortex as a consumer before it has code would be config-truth, not runtime-truth.
3. **The ordering stands.** Surface (this) -> one bridge (goal producer reads the competition,
   falsifiable against `goal_matched_no_loop` at 72.4%) -> state machine, kicked off from
   cortex, only for durable/resumable workflows per the two prior decisions. LangGraph is not
   a dependency of this repo today (`README.md:1054` names it as a direction, nothing imports
   it); it should not become one before step 2 moves a number.
