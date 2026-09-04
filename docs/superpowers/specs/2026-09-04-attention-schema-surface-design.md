# The attention schema as a surface, not a seam

**Date:** 2026-09-04
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

There is a second prize. Orion's substrate attention has taken the `bottom_up_salience`
branch 19,408 times out of 19,408 over seven days and has never once recorded a voluntary
override. Curiosity's own selection code says, in its docstring, that Orion chooses. If
that is true, **voluntary attention may already exist in Orion, in the one place
structurally incapable of reporting it** — and we have been reading the wrong lane and
concluding it never happens.

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

Not the surface. **The discriminating fields on the existing reducer, first.**

`top_down_override` has four gates it must clear (a goal exists; the goal flipped the
winner; the winner has a candidate action; the broadcast lane was fresh). The stored
self-model records `voluntary_override = None` and **nothing about which gate stopped it** —
no goal-present flag, no `effort_budget_used`. So the 19,408/19,408 result cannot currently
be root-caused, only guessed at. That is the failure mode already in the record as *"an
aggregate that cannot name a cause will hide one."*

Reasons to do this before the surface, not after:

1. It is small, self-contained, and changes no behavior — four fields on an existing row.
2. It answers whether the substrate lane's override branch is *dead* or merely *quiet*,
   which materially changes what the surface is expected to reveal. Building the surface
   first means building it without knowing that.
3. If the answer is "no goal is ever present," that is a finding about O2/O3 worth having
   on its own, independent of whether this surface is ever built.

Then, in order: settle Missing Question 1 by reading a real curiosity run; write the schema
and the substrate adapter (the cheapest, since the data already exists); reverie second;
curiosity last, since it carries both open questions.

Deferred and explicitly not bundled: the `SUBSTRATE_ATTENTION_SELF_MODEL_LOG_RETENTION_HOURS`
bump (168 → 8760), which is an operator change owned by Juniper, and the standing fix to
the `narrative_diversity` claim's "ever" question.
