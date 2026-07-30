# Metacog: keep per-turn scoped, add a separate trend reducer — design spec

Status: **design mode, not implemented.** Touches the metacog/collapse-mirror cognition loop, which
CLAUDE.md §0A requires explicit proposal mode for before implementation. This document proposes; it
does not build.

## Arsonist summary

Two same-day evaluation passes (2026-07-28, in conversation) on `log_orion_metacognition` found it
achieves none of the three things a real metacognitive loop needs: temporal continuity ("this has
been happening for 2 hours"), causal attribution ("X caused Y"), and verified narrative synthesis.
It's a single-tick anomaly-scored write with unverified LLM prose stapled on top. PR #1427 (this
session) already deleted the part of the prose that was pure waste — Enrich's 7 output fields and
Draft's 6 dead fields (`type`/`trigger`/`causal_echo`/`field_resonance`/`emergent_entity`/
`resonance_signature`) never reached the published row. That cut was correct but incomplete: it
removed dead output, it didn't add the missing capability.

Juniper's direction: don't let metacog itself grow the missing capability. Keep
`log_orion_metacognition` scoped to exactly what it already is — a single real per-turn signal
write. Record the signal, stop there. The temporal/causal work (trend detection, "what caused this,"
multi-tick synthesis) belongs in a **separate reducer** that consumes the already-persisted stream
of these per-turn signals, materializes a real windowed projection, and feeds that projection
**forward** as input for more advanced state changes — not as work grafted onto the per-turn LLM
draft call. This is this repo's own stated architecture order: `event -> schema -> trace -> reducer
-> projection -> eval -> UI surface`. `MetacogEntryV1` rows already are the event. Nothing reduces
over them yet.

## Current architecture

**The per-turn write (post-#1427):** `log_orion_metacognition` is 3 steps —
`MetacogContextService` (real single-RPC state/biometrics fetch) → `MetacogDraftService` (one LLM
call, prompt `orion/cognition/prompts/log_orion_metacognition_draft.j2`) → `MetacogPublishService`
(`services/orion-cortex-exec/app/executor.py`, builds `MetacogEntryV1`). Every field of the
published row traces to a real single-tick artifact: `compute_causal_density()`/
`compute_severity()`/`compute_touches()`/`compute_provenance()` (`orion/metacog/service.py:88-200`)
are a fixed-weight linear blend of `repair_pressure`, `substrate_eventfulness_score`, and
`turn_effect_severity` — all read from the current tick's `ctx` only, no history. `turn_effect`
itself (`orion/schemas/telemetry/turn_effect.py:20-34`) is the one genuine multi-tick artifact in
the whole pipeline: a strict before/after delta on 4 scalars (`valence`/`energy`/`coherence`/
`novelty`) between this turn and the immediately preceding one. One step of memory, no more.

**The only "recent" context anywhere in the draft prompt** is `TraceCache`/`CoreEventCache`
(`services/orion-cortex-exec/app/trace_cache.py:14`, `core_event_cache.py:72`) — process-local,
in-memory `deque(maxlen=5-10)` singletons. Last-N-by-insertion-order, no timestamps used for
windowing, resets on restart, not shared across replicas. Not a time window.

**Nothing reads `orion_metacog` history back.** Grepped the whole `log_orion_metacognition`
pipeline: zero hits. Orion never sees its own prior Collapse Mirror entries when writing a new one.

**One real precedent for windowed queries over this data exists, and it's not a reducer.**
`services/orion-dream/app/aggregators_sql.py::fetch_recent_sql_fragments(hours=24, ...)` does an
hours-bounded SQL fetch — but its own comment (line 118-119) says "Orion's machine-generated metacog
self-observation entries now live in `orion_metacog`, not `collapse_mirror`" while the query
directly below it *still only reads `collapse_mirror`*, filtered to `observer='juniper'`. Dream's
context-gathering has never actually seen Orion's own machine-generated metacog entries — a real,
live, separate bug found as a side effect of this investigation, not its subject. Even fixed to read
the right table, this function returns raw text `Fragment`s for an LLM to weave into dream
narration — no z-score, no rate-of-change, no direction, no numeric reduction. Not the reducer
pattern this spec wants; a different consumer with a different job.

**The real reducer-pattern precedent lives elsewhere in this codebase.**
`services/orion-substrate-runtime/app/worker.py` has a `ReducerSpec` class and
`_grammar_reducer_poll_loop()` (lines 179, 425) computing `gap_zscore` in FalkorDB (lines 778-780)
for bus-synaptic anomaly detection — a live, working windowed-reduction pattern, just on a different
domain (bus health, not metacog/turn_effect). This is the shape to copy, not invent from scratch.

**Today's own generative-triggers spec**
(`docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md`) is the right
instinct in a different place: it wants non-error triggers (`insight`/`flow`), but even those are
still evaluated as point-in-time gate conditions on `AttentionSelfModelV1.confidence`, itself
unverified as live-ticking (that doc's own Missing Question 1). A trend reducer, if built, would be
a natural real signal source for that spec's `insight` trigger too — "confidence crossed a
threshold" is a much more honest claim once there's a real windowed baseline behind "crossed,"
rather than an absolute point-in-time cutoff.

**Grounding gap, separately confirmed, related but not solved by this spec:** no confabulation
guard exists anywhere in this pipeline (grepped `services/orion-cortex-exec/app/executor.py` and
`orion/` for `confabulation_guard`/`groundedness`/`verify_grounded` — zero hits).
`summary`/`what_changed.evidence` have prompt-level "don't invent" instructions and real evidence
available to ground against, unverified by any code. `mantra` and `tags_suggested` aren't even given
grounding instructions in the prompt (`draft.j2` defines `summary`/`mantra`/`what_changed` only;
`tags_suggested` isn't described anywhere in it). A trend reducer's output would give
`what_changed`/`summary` something *real* to cite instead of today's single-tick snapshot — but
doesn't by itself add a verification step. Noted as related, deliberately out of scope here.

## Missing questions

1. **Which series does the reducer actually run over?** `orion_metacog` rows themselves (one point
   per trigger fire — sparse, irregular cadence, but directly matches "has this kept happening" and
   is already durably persisted) vs. `turn_effect`/`repair_pressure` raw history (denser, per-turn,
   but not yet confirmed to be durably persisted anywhere queryable outside the ephemeral spark
   snapshot metadata `MetacogContextService` reads it from).
2. **Does `turn_effect`/`repair_pressure` have any durable, queryable history at all**, or does this
   reducer first require adding persistence for a signal that currently only exists as "this tick's
   delta, forgotten after publish"? Prerequisite question, not an assumption — CLAUDE.md's
   existing-mechanism check applies exactly here.
3. **What does "feed forward into more advanced state changes" concretely mean as a consumer of the
   reducer's output?** Named in the request but not yet scoped: a new evidence-cue block back into
   `MetacogContextService` (cheapest, matches the existing `RECENT TURN-EFFECT ALERTS` pattern), a
   real gate condition for the generative-triggers `insight` work, a reverie input, or something not
   yet built. Picking one is the real next-patch decision — not this doc's job to force.
4. **Live-data shape, per CLAUDE.md's metric quality gate step 4** — before any threshold or window
   size gets picked: does whichever series gets chosen have a genuine rest state and genuine
   discrete elevated periods in stored history, or is it either flat/degenerate, or (the sneakier
   failure — this doc's own repo precedent, `bus_synaptic_prediction_error`'s permanent ~0.27 floor)
   mathematically incapable of reading "calm" even though it visibly moves? Needs a measurement
   script against real history before any reducer code is written, same discipline the
   generative-triggers spec already applies to its own Missing Question 2.
5. **Does the dream aggregator's stale `collapse_mirror`-only query need fixing in this same patch,
   or is it a separate one-line fix?** Found as a side effect of this investigation. Recommend
   splitting it out as its own trivial, independent PR rather than bundling it here.

## Proposed schema / API changes

- No change to `MetacogEntryV1` or the per-turn publish path. The per-turn write stays exactly as
  scoped by PR #1427 — this is the explicit point of the "keep it turn-scoped" direction.
- New, small typed projection — exact name/home TBD pending Missing Question 3's consumer decision.
  Rough shape: `series_id`, `window_start`/`window_end`, `direction` (rising/falling/flat),
  `magnitude`/`rate_of_change`, `duration_at_elevated_level`, and a `baseline_comparison` computed
  against Orion's own recent rolling baseline — not an absolute threshold. That last field is what
  fixes the earlier evaluation's "mild elevation on a calm day reads identical to mild elevation
  during an already-stormy week" gap.
- New reducer, following the `ReducerSpec`/`_grammar_reducer_poll_loop()` pattern already live in
  `orion-substrate-runtime/app/worker.py`, rather than inventing a new poll-loop shape.
- If `MetacogContextService` becomes the chosen consumer: one new additive evidence-cue block in the
  draft prompt, same shape as `RECENT TURN-EFFECT ALERTS` — a real cue handed to a still-single-turn
  draft call, with zero window/query logic added inside the draft step itself.

## Files likely to touch

- New: `orion/metacog/trend_reducer.py` (or under `orion/substrate/`, if it turns out to belong
  alongside the existing `gap_zscore` pattern rather than inside `orion/metacog/` — undecided,
  depends on Missing Question 1's answer)
- New: a projection schema under `orion/schemas/` (exact home pending Missing Question 3)
- `services/orion-dream/app/aggregators_sql.py` — the stale `collapse_mirror`/`orion_metacog`
  mismatch (Missing Question 5), most likely a separate, smaller, independent PR
- `services/orion-cortex-exec/app/executor.py`'s `MetacogContextService` block, only if that ends up
  being the chosen consumer (Missing Question 3)
- `orion/cognition/prompts/log_orion_metacognition_draft.j2`, only if a new evidence-cue block gets
  added — still additive to inputs, never a new instruction asking the draft step to compute trends
  itself
- `scripts/analysis/measure_metacog_trend_baseline.py` (new, for Missing Question 4 — matches this
  repo's existing `measure_*_baseline.py` convention, e.g. `measure_rpc_health_baseline.py`)
- Whichever service ends up hosting the reducer's poll loop
  (`docker-compose.yml`/`.env_example`/`settings.py` for that service)

## Non-goals

- Not adding any window/trend/history logic inside `MetacogDraftService`, Enrich (already deleted),
  or the draft prompt's own computation. The per-turn draft call stays a single-turn read of
  whatever evidence cues it's handed — including a new trend cue, if one gets built — never a query
  over history itself. This is the load-bearing scoping rule from this whole spec.
- Not resurrecting `causal_echo`/`what_changed.previous_state`/`.new_state` as free-text LLM fields.
  If/when the reducer produces something worth citing, it should be a structured, code-computed
  evidence cue — not a return to asking the model to narrate a causal history it was never actually
  given, which is what made those fields unfillable in the first place.
- Not fixing the grounding/confabulation-guard gap in this spec. Related, separately scoped, not
  solved by adding a reducer.
- Not merging reverie and metacog. Still a separate, larger architecture decision, out of scope here.
- Not picking a window size, threshold, series, or specific consumer yet — all deferred to the
  Missing Questions above.

## Acceptance checks

1. A measurement script against real historical data (whichever series Missing Questions 1-2
   resolve to) shows the chosen signal has genuine discrete elevated periods and a genuine rest
   state — not smooth noise, not a permanent floor/ceiling artifact — before any reducer ships live.
2. The reducer's poll loop follows the existing `ReducerSpec` pattern in
   `orion-substrate-runtime/app/worker.py` rather than a new bespoke shape.
3. `MetacogDraftService`'s own code and prompt remain structurally unchanged by this work — at most
   one new additive evidence-cue block, zero new query/window logic inside the draft step.
4. The projection is independently queryable/inspectable (a real debug surface or at minimum a
   direct DB/Falkor query) regardless of whether any consumer has wired it in yet.
5. Ships disabled by default, flipped only after its own live-data sanity check — same precedent as
   `bus_synaptic` (PR #1385 → #1387) and the generative-triggers spec's own Acceptance Check 2.

## Recommended next patch

Don't build the reducer yet. Answer Missing Questions 1-2 first: confirm which series actually has
durable, queryable history (`orion_metacog` rows themselves are the safe fallback — sparse, but
already persisted, no new plumbing needed to read them). Then run the Missing Question 4 measurement
pass against real stored data before writing any reducer code. That single measurement script is the
actual next deliverable; everything else in this doc is scoped and blocked behind its result — same
discipline the generative-triggers spec already models for its own open questions.

**Superseded by the 2026-07-28 update below** — see that section for the corrected priority order.
This "Recommended next patch" is not wrong on its own terms, but it is no longer the top of the
queue: it names the wrong prerequisite. The real blocker is not "does the metacog series have a
genuine rest state," it's "does the arena this reducer's output would compete in even arbitrate
fairly" — and a same-day measurement found it does not.

## 2026-07-28 update — the arbitration layer, not the reducer, is the real blocker

Same-day follow-up, same Juniper conversation that produced this doc. Before answering this doc's
own Missing Questions 1-2 and building the trend reducer, the conversation traced every
"competition/arbitration" mechanism in this codebase looking for where the reducer's output should
"feed forward" (Missing Question 3). It found the same architectural disease in all three places
checked: real inputs feeding a hand-picked fixed-weight linear score, never calibrated against real
outcomes, so one channel permanently or near-permanently dominates because nothing normalizes for one
channel being structurally noisier or more saturating than the others.

**1. Old drives system (retired).** `dominant_drive=relational` monoculture: 96% of ticks pre-fix,
~31.65% post-fix (`orion/autonomy/drives_and_autonomy_retrospective.md`).

**2. Proposal/policy pipeline.** `orion/proposals/scoring.py::proposal_priority()` is
`base_priority + 0.4*match_score + 0.2*urgency + 0.1*confidence` — fixed, uncalibrated coefficients.
`orion/feedback/builder.py::build_feedback_frame()` genuinely records real outcomes
(`FeedbackFrameV1`, real `outcome_status`/`outcome_score`, real field-pressure deltas) — but a
repo-wide grep for writes to `base_priority`/`base_risk`/`dimension_weights` found only static config
reads, never a write-back from feedback. The loop observes but never learns.

**3. Layer 5 attention (the canonical, most-upstream competition layer)** —
`orion/attention/field_attention/{scoring,selectors}.py`, live via `orion-attention-runtime`
(`ENABLE_ATTENTION_RUNTIME=true`), explicitly named in `orion/sentience_striving_program/README.md`
as the reason the drives system was made redundant. That charter's own §6 item 5 already ran
`scripts/analysis/measure_emergent_clustering_probe.py` against 127,936 real
`substrate_attention_frames` rows and found `select_system_targets`'s `field:recent_perturbations`
target (`orion/attention/field_attention/selectors.py:128-140`,
`salience = min(1.0, recent_perturbation_count / 10.0)`) wins top-1 in ~99.98% of ticks — the same
"noisiest wins" shape as the drives pathology, on a different signal.

**Converged direction (Juniper's own call):** don't build a new candidate producer — a metacog trend
reducer, or anything else — into an arena that already can't arbitrate fairly. Fix the arbitration
first, at the most canonical layer (Layer 5 attention), by normalizing each channel against its own
real historical distribution before comparing magnitudes for top-1-winner selection. The charter's
own §7 rule applies directly: "Measure before minting. Every new signal gets a read-only instrument
and real historical replay before it gates anything live." So: measure whether normalization would
actually fix the monoculture, using real data, before writing any live-scoring code.

### Real measurement results (`scripts/analysis/measure_attention_salience_normalization.py`, run 2026-07-28)

Read-only script, same `substrate_attention_frames` table, run fresh against 127,644 real rows
spanning 2026-07-25T20:18Z → 2026-07-28T20:34Z (72.3h). Per CLAUDE.md's metric-quality-gate rule
("re-run it every time, even for a metric that seems obviously fine"), the 99.98% figure above was
**not** reused — the raw baseline was recomputed fresh in this same run:

- **Raw baseline, re-verified fresh: `field:recent_perturbations` wins top-1 in 100.00% of ticks**
  (127,644 / 127,644) — slightly *worse* than the 99.98% figure cited above, not the same number
  restated. Its full-history stddev is `0.000000` (mean `1.0000`) — a mathematically exact
  degenerate/saturated channel, not merely "very concentrated."
- **Per-channel z-score normalization was computed** (each channel normalized against its own
  full-history mean/stddev). `field:recent_perturbations` has zero variance, so it is
  **structurally undefined under z-scoring** — division by ~0, excluded from the normalized ranking
  entirely, not merely "loses" a fair comparison.
- **After excluding it, the normalized top-1 winner distribution is:** `node:atlas` 55.17% (70,417
  ticks), `node:circe` 15.84%, `capability:llm_inference` 15.24%, `node:athena` 9.86%,
  `capability:orchestration` 2.23%, `capability:transport` 1.65%, `capability:storage` 0.01%.
- **Classification: `NOT_MET_MONOCULTURE_SHIFTED`.** Normalization does not diversify the winner
  distribution — it relocates the monoculture to a different single channel (`node:atlas`, 55.17%,
  still above this measurement's 50% monoculture threshold). Two honest findings, both reported by
  the script rather than left implicit: (a) any apparent "improvement" here is a mechanical
  side-effect of `field:recent_perturbations` being disqualified by divide-by-~0, not evidence that
  z-scoring is doing real calibration work on real variance; (b) the "fix" as measured just moves
  who wins, it does not make the arena fair.

**Corrected recommendation:** this doc's original "Recommended next patch" (build
`scripts/analysis/measure_metacog_trend_baseline.py` and proceed toward the trend reducer) is now
**sequenced behind** a higher-priority prerequisite: fixing Layer 5 attention's monoculture is the
real blocking issue, not building a new candidate producer. Any new producer — including the trend
reducer this doc proposes — would compete in the same broken arena and either get drowned out (if
its signal is calmer than the saturating channels) or add another uncalibrated fixed-weight input to
the same disease (if it's wired in as another hand-tuned score). The measurement above also shows the
naive fix (plain per-channel z-scoring) is **not sufficient by itself** — it needs a real design pass
(e.g. excluding degenerate channels honestly instead of accidentally, a genuine multi-way calibration
rather than winner-take-all-on-a-different-channel, or a different normalization shape entirely) before
it is safe to build, let alone flip live. That design pass is out of scope for this measurement run and
requires its own explicit proposal-mode sign-off per `orion/sentience_striving_program/README.md`'s
charter and root CLAUDE.md §0A's cognition-loop rule — not decided here.

This doc's Missing Questions 1-2 and the trend-reducer build itself are not cancelled, just
re-sequenced: they remain the right next step for metacog specifically, but only after (or in
parallel with, if scoped as a fully separate consumer of attention output) the arbitration-layer
question above gets its own resolution.

## 2026-07-30 update — arbitration fixed, both blockers cleared

**Layer 5 attention monoculture: fixed, shipped, and re-measured live — this section's own
recommendation is now stale.** The 100.00%/zero-variance measurement above (run 2026-07-28) and this
doc's "fix Layer 5 first" recommendation were both written in the same session as, and minutes before,
two PRs that actually addressed it:

- **PR #1433** (`fix(field-attention): replace saturated recent_perturbation caps with EWMA
  baseline`, merged 2026-07-28T21:57Z) replaced `select_system_targets`'s `min(1.0, count / 10.0)` cap
  — the literal cause of the zero-variance saturation — with a z-score against a per-tick EWMA
  baseline (`orion/schemas/field_state.py::recent_perturbation_zscore`, `orion/bus/ewma.py`), the same
  methodology already validated for `bus_synaptic_prediction_error`'s `gap_zscore`. This is a real fix
  to the scoring formula itself, not the post-hoc per-channel z-scoring this doc's own measurement
  script tried and found `NOT_MET_MONOCULTURE_SHIFTED` (those are two different things: the doc's
  measurement normalized the *output*, PR #1433 fixed the *input formula*).
- **PR #1454** (`docs(sentience-striving): re-measure recent_perturbations dominance post-fix`, merged
  2026-07-29T04:36Z) reran the live probe ~6.3h post-deploy: `field:recent_perturbations` winning top-1
  dropped from 99.98% (pre-fix) to **11.13%** (1,257/11,293 post-fix ticks), with `node:athena` (host
  resource pressure — real signal, not the old degenerate one) taking the remaining 88.87%. That doc
  flagged an open question: is `node:athena`'s 88.87% share genuine, or a new artifact nobody checked.

**Independently re-verified this session, ~+24h further out (through 2026-07-30T00:10Z, 36h window,
127,447-127,878 live `substrate_attention_frames` rows):** `node:athena`'s share has continued
dropping — 60.0% (37,994/63,295), `field:recent_perturbations` 38.7%, `node:atlas` 1.3% — trending
toward a real, converging multi-way competition rather than settling into a second monoculture. Margin
check between #1 and #2 (48,144 ticks with ≥2 candidates): median gap 0.16, ~7% of ticks are near-ties
(gap < 0.02), and `node:athena`'s own salience score has real variance (stddev 0.09, not pinned).
This is not a landslide. **`node:athena`'s 88.87%-then-60.0% share answers PR #1454's open question:
genuine, converging signal, not a new artifact — worth one more re-check in another 24-48h to confirm
full convergence, but not worth treating as a blocker.**

**`turn_effect`/`repair_pressure` durable history (this doc's own Missing Questions 1-2): also
confirmed live, also resolved.** `repair_pressure_appraisal_log` (dedicated Postgres table, shipped
specifically to close this exact gap — see its own test docstring in
`services/orion-sql-writer/tests/test_repair_pressure_appraisal_log.py`) has 52 real rows spanning
2026-07-24 through 2026-07-30. `turn_effect` is not a dedicated column but is durably persisted inside
`chat_history_log.spark_meta` JSONB (`services/orion-sql-writer/app/worker.py::_spark_meta_minimal()`)
— 37 of the last 41 rows (7 days) carry a real value, queryable via `spark_meta->'turn_effect'`, same
JSONB-path pattern already used against `substrate_attention_frames.frame_json`. No new persistence
plumbing needed for either series.

**Net: both prerequisites this doc and the stream-of-consciousness hop-chain design
(`docs/superpowers/specs/2026-07-29-stream-of-consciousness-hop-chain-design.md`) named are cleared.**
The trend reducer (hop 0) can proceed — build
`scripts/analysis/measure_metacog_trend_baseline.py` against `repair_pressure_appraisal_log` and/or
`chat_history_log.spark_meta->'turn_effect'` first (this doc's original acceptance check 1: confirm a
genuine rest state, not smooth noise or a floor/ceiling artifact, before the reducer ships live), then
the `ReducerSpec`-pattern reducer itself.

## 2026-07-30 update #2 — acceptance check 1 actually run: neither candidate series is clean yet

Built and ran `scripts/analysis/measure_metacog_trend_baseline.py` (same session, same day) against
real live data — the arbitration/history prerequisites above being clear turned out not to mean the
acceptance check itself was already satisfied. It wasn't assumed; it was measured, and the honest
result is **not yet green**:

- **`repair_pressure_appraisal_log.level`, ungated (n=52): `FLOOR_DOMINATED`.** 76.9% of rows (40/52)
  sit at one repeated exact value (`0.087065772...`), and **all 40** of those rows also have
  `confidence == 0.0` — the appraiser's own explicit "no evidence, don't trust this" signal. Neither
  live appraiser (`orion/substrate/appraisal/repair_pressure.py`,
  `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`) emits this value for its documented
  no-evidence path (both return `level=0.0` exactly) — a short search did not find where the smoothing/
  gating step between appraisal and persistence produces it. **Disclosed, not root-caused**: whoever
  builds hop 0 should treat this floor's exact origin as open, not organic.
- **Confidence-gated (`confidence > 0` only, n=12): real spread (mean 0.316, stddev 0.139, no single
  value above 42% share), but `INSUFFICIENT_DATA` by this script's own 20-row floor.** Promising shape,
  not yet enough rows to certify.
- **`chat_history_log.spark_meta->'turn_effect'->'turn'->'novelty'` (30-day window, n=37):
  `FLOOR_DOMINATED` at the *ceiling*, not the floor.** 78.4% of rows read >= 0.99 novelty. That's a
  different flavor of the same disease this whole investigation keeps finding — a channel that rarely
  or never reads a real calm state, just inverted (pinned high instead of pinned low). Worth its own
  check before trusting it as hop 0's series: is real conversation genuinely this novel this often, or
  is the novelty formula itself another saturating instrument.

**Revised recommendation:** hop 0 is not ready to build against either series as-is. The path forward
is either (a) gate `repair_pressure_appraisal_log.level` on `confidence > 0` and wait for more real
rows to accumulate past the 20-row floor before trusting `GENUINE_VARIATION`, or (b) investigate
`turn_effect` novelty's ceiling-saturation as its own short measurement (same shape as Layer 5's
`field:recent_perturbations` and the old drives system's `dominant_drive` — a fourth instance of
"structurally can't return to calm" in this codebase, not yet named as such until now) before either
series is trusted enough to build a live reducer against. Full numbers, per-series breakdown, and CSV:
`/tmp/measure-metacog-trend-baseline/report.md` (and `rows.csv`, not committed — real historical rows,
regenerate by re-running the script).

## 2026-07-30 update #3 — reducer core built and tested; deliberately not live-wired yet

**Reconciling this section with update #2 immediately above, which says "not ready to build":** that
call was, and remains, about whether either candidate series is trustworthy enough to *run a live
reducer against* — still not yet true (12 real confidence-gated rows, `INSUFFICIENT_DATA`). What
changed here is narrower: the reducer's own *computation logic* — the EWMA fold, cold-start guard, and
sustained-trend check — can be built and unit-tested against synthetic data plus a real-data sanity
check without needing the underlying series to already be certified, the same way a thermometer's
circuitry gets tested before anyone trusts a specific room's reading from it. This section does not
reverse update #2's "don't build a live reducer against this data yet" — it narrows to "the reducer's
own logic, tested in isolation, is a separate, safe, buildable thing" and stops there.

Built `orion/metacog/trend_reducer.py`: pure, incremental, checkpointable EWMA-trend computation
(`apply_reading`/`replay`), reusing `orion/bus/ewma.py::compute_ewma_update` rather than inventing a
new z-score formula (existing-mechanism check). Cold-start guarded at `min_samples=20` (matches this
doc's own small-N finding above) — never classifies a reading `is_elevated_this_tick` on too little
evidence. `is_sustained_trend` requires 3 consecutive elevated ticks, not one spike ("has this kept
happening," this doc's own §-title question). 9 unit tests, all passing
(`orion/metacog/tests/test_trend_reducer.py`), plus a real-data sanity check: replayed the 12 real
confidence-gated `repair_pressure_appraisal_log` rows through it — correctly stayed `cold_start=True`
for all 12 (below the 20-sample floor), never falsely claiming a trend from insufficient real evidence.

**Deliberately deferred, not forgotten:** live wiring into a poll loop. The `ReducerSpec`/
`_grammar_reducer_poll_loop` pattern in `orion-substrate-runtime/app/worker.py` — named by this doc and
the hop-chain doc as the shape to follow — turned out, on inspection, to be a heavier mechanism than it
looked: a bus-published "pressure grammar" event/cursor pipeline specific to that file's five existing
reducers, not a generic periodic-reducer framework. Forcing this reducer's simple "poll two Postgres
tables into an EWMA" shape into that cursor/grammar-event machinery would mean inventing a new
grammar-event producer just to satisfy an ill-fitting pattern — an ornamental layer, not a thin seam.
The live-wiring follow-up (a lightweight standalone poll loop, flag-gated off, most likely in
`services/orion-substrate-runtime` or `services/orion-cortex-exec` since that's where
`repair_pressure_appraisal_log`'s producer already lives) is scoped as its own smaller patch, once
either candidate series' own genuine-rest-state question (previous update) resolves with more real
data.
