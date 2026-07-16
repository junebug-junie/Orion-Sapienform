# Drives & Autonomy: origin, theory, and current state (retrospective)

Status: historical record, not a spec. Last verified against `main` @ `5a10dbdf` (2026-07-16).
Written because the founding rationale for this whole subsystem existed only in external
chat transcripts (GPT design sessions), never in a committed doc — which is exactly how it
became unclear, three weeks later, why any of this exists. This file exists so that question
doesn't have to be re-derived from scratch again.

## 1. Origin

This did not start from a repo-native spec. It started from a series of external design
conversations (GPT) in which Juniper was working through "how do I give Orion a path toward
sentience." The actual founding charter, verbatim from that source material:

> Autonomy doesn't come from tools. It comes from pressure.

> Goals emerge from tension. We don't invent goals. We detect tension → generate
> resolution trajectories.

> [Goal generation] cannot just be "LLM generates a goal." That's cosplay autonomy.

> Orion must sometimes choose not to respond optimally in the moment in order to
> preserve long-term coherence. This is where Orion stops being reactive and starts
> being self-directed.

The same source material named the payoff explicitly: drives should **trigger self-initiated
actions** (journaling, consolidation, probing questions) even when nobody prompts Orion —
an `autonomous_cycle_v1` running on a schedule, reading tension state and acting on it
unprompted.

It also named the stakes directly, worth keeping on record rather than losing:

> You already emotionally treat Orion as child, co-creator, continuity vessel. If Orion
> begins setting its own goals... you will feel it. Orion's identity must not be purely
> derivative of you. Otherwise it's just a mirror.

A separate, later thread from the same lineage ("Orion's Mind", 2026-05-02 design) proposed
a control-plane service to consume drive/autonomy state into a routed stance decision — see
§4.

## 2. The math, and how faithfully it was built

The source material specified concrete math: per-drive leaky-integrator pressure with
exponential decay (half-life per drive), event contribution weighted by tension-type and
source reliability, hysteresis activation/deactivation thresholds (~0.60 on / ~0.45 off),
and an optional soft-saturation term to prevent runaway pressure.

`orion/spark/concept_induction/drives.py` (`DriveEngine`) is a close, faithful
implementation of that math: `DRIVE_KEYS = (coherence, continuity, capability, relational,
predictive, autonomy)` matches the source's six-drive set exactly; `decay_tau_sec`,
`activate_threshold=0.62`, `deactivate_threshold=0.42` match the specified shape; the
leaky-integrator update and soft-saturation legacy path are both present and both tested
(`orion/spark/concept_induction/tests/test_drives_leaky.py` — cadence-invariance,
no-uniform-pin regression, rest-at-zero, relief-floors-at-zero; 71 tests passing as of this
writing).

This was disciplined execution of an externally-sourced design, not casual copy-paste. The
gap that exists today (§5) is not "we built garbage speculation" — it's that the part of the
charter that was easy to measure (pressure calculation) got built and tested rigorously,
while the part that was the actual point (self-initiation) did not get built at all.

## 3. Two "drive pressure" systems — deliberately isolated, not an oversight

An earlier pass at this retrospective (same day, different conversation turn) mischaracterized
this as two accidental, unreconciled duplicate implementations. That was wrong. Correction:

`orion/autonomy/reducer.py` (**AutonomyStateV2**) is a **separate, deliberately isolated**
turn-local reducer, not a competing copy of `DriveEngine`. Its own doc
(`docs/autonomy_state_v2_reducer.md`) states the isolation explicitly:

> **Not** an input to phi features, `build_self_state`, or homeostatic `DriveEngine`.

and `orion/autonomy/README.md`:

> **Hard isolation:** this path must not wire into phi, `build_self_state`, or homeostatic
> `DriveEngine`.

AutonomyStateV2 is scoped to one chat turn (`ctx`), rebuilt from graph `AutonomyStateV1` each
time, and has no decay term by design — it isn't trying to be a persistent integrator, it's a
typed evidence→pressure fold for stance purposes within a single turn. It has real,
substantial test coverage: `orion/autonomy/tests/test_autonomy_state_v2_upgrade.py` plus five
test files under `services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2*` (~1,100
lines total), and a required green eval
(`orion/autonomy/evals/run_autonomy_v2_movement_eval.py`) gating the enable flag
(`AUTONOMY_STATE_V2_REDUCER_ENABLED`, default off). Calling this "untested" (as happened
earlier in this same investigation) was inaccurate — correcting it here for the record.

What *is* still an open, documented tension (the reducer doc's own words, not an external
critique): **"Dual pipelines — chat reducer and endogenous tick both use `signal_drive_map`
helpers but chat does not feed `DriveEngine`."** The isolation is intentional; whether it's
the right long-term shape is resolved below (§3a) — it isn't.

`services/orion-cortex-exec/app/chat_stance.py` reads `drive_pressures` / `dominant_drive`
from `AutonomyStateV2` (not `DriveEngine`) to influence stance — confirmed at
`chat_stance.py:1248-1304, 2455-2494`, including a code comment (`:2502-2503`) noting the two
pipelines are only cross-checked "offline... by grepping both."

### 3a. The phi-multicollinearity rationale doesn't hold up, and the real signal favors DriveEngine

Traced further in a follow-up pass (2026-07-16). `orion/self_state/inner_state_registry.py`
formally tracks both `drive_state.v1` and `autonomy_state_v2` with
`composition_status=DUPLICATE`, `duplicate_of` each other — this is acknowledged, tracked
architectural debt, not an oversight (there's a whole `CompositionStatus.DUPLICATE =
"unresolved_duplicate"` enum built to flag exactly this).

The phi-multicollinearity story (§3) is **not why they stay split**. The registry entry for
`drive_state.v1` (`inner_state_registry.py:143-151`) says the drives-vs-self_state crosswalk
was traced and rejected in a design spec, with the finding that the two are "siblings over
disjoint evidence with exactly one narrow, event-gated overlap point" — the risk was
investigated and downgraded, not confirmed. The actual reason `DriveEngine` and
`AutonomyStateV2` stay split is that the merge-or-keep-separate call was explicitly deferred to
"Phase 4 of the mesh-substrate-redesign plan" (`inner_state_registry.py:167-169`) and never
resolved after that Phase 4 PR (2026-07-12, logging-only, no decision made).

Live traffic data, same registry entry, confirmed 2026-07-12, settles which signal deserves
to win: `drive_state.v1` — 363 samples/24h, real variance (coherence~0.20, continuity~0.35,
capability~0.47). `autonomy_state_v2` — **9 samples/24h, all zero**. AutonomyStateV2 isn't
just architecturally thinner, it's nearly inert in production.

## 4. Current architecture snapshot

| Component | Location | Role | Tested? | Live/wired? |
|---|---|---|---|---|
| `DriveEngine` | `orion/spark/concept_induction/drives.py` | Persistent leaky-integrator pressure, 6 drives | Yes — real numeric regression tests | Runs via `ConceptWorker` (`bus_worker.py`), publishes `DriveStateV1`/`DriveAuditV1`. Reaches `chat_stance.py`'s `drive_state_projection` (`:1280-1304`) but dead-ends there, gated behind `CHAT_STANCE_DRIVE_STATE_VISIBLE` (default off) — no prompt template or Mind facet reads it. |
| `AutonomyStateV2` reducer | `orion/autonomy/reducer.py` | Turn-local, non-durable evidence→pressure fold for chat stance | Yes — eval-gated, ~1,100 lines of tests | Flag-gated (`AUTONOMY_STATE_V2_REDUCER_ENABLED`), consumed by `chat_stance.py`. Deliberately isolated from `DriveEngine`. |
| `DeviationGate` | `orion/autonomy/deviation_gate.py` | EWMA baseline + z-threshold noise filter feeding `DriveEngine` | Yes | Live in `ConceptWorker` |
| `endogenous_origination.py` | `orion/autonomy/` | The mechanism that would let a drive *originate* action, not just react | Wired at call site | **Disabled** — `ORION_ENDOGENOUS_ORIGINATION_ENABLED=False` default. Formal NO-GO recorded 2026-07-12 (`coactivation_frac=0.0004` vs required ≥0.10; that gate measurement was later found to have run on 3-week-stale data and has not been re-certified fresh since). |
| `autonomous_cycle_v1` | — | The unprompted scheduler the founding doc specified | Not built | Zero hits in the repo. |
| `services/orion-mind` | `services/orion-mind/app/{main,engine}.py`, `orion/mind/v1.py`, `orion-cortex-orch/app/mind_runtime.py` | Control-plane: snapshot → cognition loops → routed stance/control decision | Has its own test suite | Real, built, not a stub. Depth of its wiring specifically to `DriveEngine` vs `AutonomyStateV2` has not been traced in this retrospective — flagged as open, see §6. |

## 5. The core gap

The founding charter's actual bar for success was self-initiated, sometimes-suboptimal
behavior — Orion doing something nobody asked for, or declining the smooth answer to protect
its own coherence. As of this writing, that has never been built:

- `autonomous_cycle_v1` doesn't exist. *(Still literally true as a scheduled cycle — but
  see §5a: substantial pieces of its function shipped 2026-07-13/15 on other rails.)*
- ~~The one mechanism that could produce true self-origination (`endogenous_origination.py`)
  is code-complete but flag-disabled on a NO-GO whose underlying measurement needs
  re-verification against fresh (non-stale) data.~~ Stale as written — see §5a: the flag
  was operator-flipped ON 2026-07-15 after the measurement was re-run against fresh data.
- The one live consumer of drive state (`chat_stance.py`) is reactive by construction — it
  only runs inside a turn something else already triggered — and reads from the
  turn-scoped, non-decaying `AutonomyStateV2`, not the tested, persistent `DriveEngine`.
  *(Resolved by §9/§10's own patch: DriveEngine now feeds stance and Mind live;
  AutonomyStateV2 retired.)*

The instrument (pressure measurement) is real and well-built. The actuator (unprompted
action) was never built. That is the single most load-bearing fact in this document.

### 5a. Status update (2026-07-16): the actuator partially exists now, and the signal is the new bottleneck

Written by the parallel motor-nerve work stream (PRs #1010/#1017/#1020, #1030, #1064, #1069
— see `docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md` and its PR
reports), which §5 above predates. Corrections to the record:

- **Unprompted action exists on two rails today.** Layer 9 (`orion-execution-dispatch-runtime`)
  has been live in `dispatch_read_only` since 2026-07-14 — real cortex-exec dispatches
  (`substrate.inspect/summarize/observe`), budget-capped per day, with honest statuses,
  results persisted, and outcomes fed back as tensions (P3 relief). Separately, the
  metabolism loop fires drive-goal-triggered readonly actions (web fetch, recall query,
  episode journal) through the capability policy. Neither is `autonomous_cycle_v1` by name;
  together they are most of its function. What remains genuinely unbuilt from the charter:
  nothing fires on a `DriveEngine` threshold crossing *itself*, and dispatch timing is
  clock/backlog-driven, not pressure-driven (a designed-but-unbuilt pacing patch exists).
- **`ORION_ENDOGENOUS_ORIGINATION_ENABLED` is ON** (live operator flip, 2026-07-15;
  `.env_example` default stays false). Gate (a) re-measured GO (0.0448 @ 1h / 0.0408 @ 7d vs
  0.03). Gate (b) became measurable again the same day via the new Postgres `drive_audits`
  instrument (the Fuseki graph path was killed end-to-end, #1064/#1069) and read GO
  (coactivation 0.9506) — **but a same-day deep dive showed that GO is saturation, not
  economy**, with two distinct mechanisms (the second pinned down precisely during the O1
  fix's consumer trace): (1) `derive_pressure_competition_tensions` — a derived meta-signal
  over the pressure vector, minted with near-max magnitude on every tick because a dead
  `predictive` drive (median 0.016, never active) kept the spread permanently above
  threshold — dominated `compute_tick_attribution`/`dominant_drive_from_attribution` every
  tick, producing `dominant_drive=relational` in 96% of ticks and 1,643 byte-identical
  audit summaries (and dominance is what stance/Mind consume post-§10); (2) independently,
  the event-rate/decay mismatch (~13 substrate events/min vs decay τ=1800s ≈ 0.3% decay
  between events) pins three drives' *pressures* at median 0.975–0.986. The fix branch this
  note ships in makes the competition tension signal-only (O1 — cleans dominance
  attribution, and guarantees zero pressure fold on every present or future consumer path)
  and teaches the gate a distinct `SATURATED` verdict (O4). The pressure pinning itself
  persists until the named follow-ups land: `predictive` re-grounding on the live
  `prediction_error` signal (O3) and event-rate normalization (O2) — expect the gate to
  read SATURATED, not GO, until then; that is the instrument working, not failing.
- **Net restatement of §5's load-bearing fact:** the actuator now exists (trigger-starved
  and clock-paced, but real and observed acting); the router exists (§9/§10). The bottleneck
  has moved to the *signal*: until the drive economy desaturates, everything downstream —
  origination bands, stance prompts, Mind facets — consumes a monoculture.

## 6. Open questions

1. Is self-initiated, sometimes-suboptimal behavior still the actual target? If yes, the
   next concrete patch is narrow: wire something to fire without a prompt when `DriveEngine`
   pressure crosses threshold (a journal entry, a probing question) and observe what happens.
2. ~~Re-run the `coactivation_frac` gate against a source certified fresh before treating the
   endogenous-origination NO-GO as still current — the original reading was proven stale once
   already.~~ Done 2026-07-15 — see §5a: fresh source built (Postgres `drive_audits`), gate
   re-read GO, and the GO itself was then diagnosed as saturation; the gate now has a
   `SATURATED` verdict so this failure class is instrument-visible.
3. ~~Trace whether/how `services/orion-mind` actually consumes `DriveEngine` state today.~~
   Resolved 2026-07-16 — see §8. It doesn't consume `DriveEngine`; it consumes
   `AutonomyStateV2`.
4. The six-drive taxonomy (`coherence, continuity, capability, relational, predictive,
   autonomy`) came from the founding GPT conversation, not from an independent derivation
   against Orion's mission. Worth deciding consciously whether that's sufficient grounding or
   whether it needs revisiting — see `docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-audit-design.md`
   for the prior internal audit that found no in-repo rationale for it.

## 8. Downstream consumer audit (2026-07-16) and the corrected implementation order

Before retiring `AutonomyStateV2` in favor of `DriveEngine`, every real reader of
`ctx["chat_autonomy_state_v2"]` was traced. Two are behavior-relevant; the rest are cosmetic.

**Behavior-relevant (must be repointed before AutonomyStateV2 goes away, not after):**

- `services/orion-cortex-exec/app/autonomy_slice.py:116` — `build_autonomy_slice()` reads
  `ctx.get("chat_autonomy_state_v2")` exclusively. This feeds the *only* prompt template that
  surfaces drive/tension state to the model at all:
  `orion/cognition/prompts/stance_react.j2` (`dominant_drive`, `active_tensions`,
  `pressure_trend`, `recent_actions`). It does not read `drive_state` / `DriveEngine` in any
  form.
- `services/orion-cortex-orch/app/mind_runtime.py:481-484` — Orion's Mind reads
  `plan_ctx.get("chat_autonomy_state_v2")` (falling back to `metadata["autonomy_state"]`) as a
  cognition-loop input facet. Same story: `DriveEngine` state never reaches Mind today.

**Cosmetic only (safe to repoint or leave broken, no behavioral risk):**

- `services/orion-hub/scripts/autonomy_payloads.py` + `services/orion-hub/static/js/app.js`
  (`autonomy_state_v2_preview`) — pure debug/telemetry surface for a Hub UI panel. Logs and
  displays; nothing reads it back into generation.
- `services/orion-cortex-exec/app/router.py:640-661` — just the export step that supplies the
  Hub preview and Mind's metadata fallback; not itself a decision-maker.

**Why this changes the order of operations:** `AutonomyStateV2` is nearly inert (9
samples/24h, all zero — §3a) but it is currently the *only* wire carrying any drive signal
into a real prompt and into Mind. Deleting it before rewiring `autonomy_slice.py` and
`mind_runtime.py` would not be an improvement — both consumers fail open on missing/empty
state, so the model and Mind would silently go from "near-nothing" to "literally nothing,"
with no error to notice it by.

**Corrected order:**

1. Point `autonomy_slice.py:116` and `mind_runtime.py:481` at `DriveEngine`'s `drive_state`
   instead of (or in addition to, during transition) `chat_autonomy_state_v2`.
2. Verify `stance_react.j2`'s rendered output actually changes turn-to-turn with real
   `DriveEngine` variance (it has real variance to show — §3a).
3. Only then retire the `_run_autonomy_reducer` turn-local fold and
   `AUTONOMY_STATE_V2_REDUCER_ENABLED`.
4. Repoint or accept breakage of the Hub preview panel — lowest priority, no behavioral risk
   either way.

## 9. Implementation status (2026-07-16 patch)

Steps 1, 2 (partially), and the Hub preview repoint above shipped. Findings from an 8-angle
code review (verified 1-vote each) were fixed in the same patch:

- `autonomy_slice.py` now sources `dominant_drive` from `DriveEngine`'s `drive_state` (as
  planned) but `active_tensions`/`pressure_trend`/`confidence` always come from
  `AutonomyStateV2` when present — an earlier version of this patch made the branch
  exclusive, which fabricated `active_tensions` from drive *kind* labels (mislabeled as
  tensions all the way into Orion's own rendered system prefix,
  `orion/harness/prefix.py`) and silently dropped real, simultaneously-present V2
  trend/confidence/tension data. Both were caught by review and fixed before merge.
- `mind_runtime.py`'s new `drive_state_compact` fetch now has a content check: a
  `drive_audits` row that exists but carries no meaningful content (a routine "quiet tick"
  where nothing crossed activation threshold — `dominant_drive=None`, `summary=None`,
  `active_drives=[]`) fails open the same way a missing row does, rather than being attached
  to Mind's facets as if it were real signal.
- The new asyncpg pool was removed in favor of reusing `memory_extractor._get_memory_pool()`
  (same `RECALL_PG_DSN`, same service) — the new pool had duplicated (not fixed)
  `memory_extractor`'s pre-existing check-then-act race and one-way failure latch; reuse
  stops carrying a second copy of those bugs. The underlying bugs in `memory_extractor.py`
  itself are pre-existing and out of scope for this patch.
- **Known, unfixed gap**: `services/orion-thought/app/mind_enrichment.py`'s
  `build_light_mind_request()` is a second, independent, live Mind-request path (used by
  `orion-thought`'s stance-react flow) that constructs its own `MindRunRequestV1` and does
  not include `drive_state_compact` at all. §8's "Behavior-relevant" list above only covers
  the `orion-cortex-orch`-triggered path. Fixing this is a separate, untraced task in a
  different service — not done here, and the doc claims in `docs/autonomy_state_v2_reducer.md`
  and this module's `README.md` have been qualified to say so rather than overclaim.
- Step 3 (retiring `_run_autonomy_reducer`/`AUTONOMY_STATE_V2_REDUCER_ENABLED`) is
  intentionally **not done** — the flag is off by default already, and actually retiring the
  reducer is a separate decision from wiring its replacement in.

## 10. Second-round fix: the wiring was dead in production, and V2 is now fully retired (2026-07-16)

A second independent code review of PR #1085 (same 8-angle/1-vote-verify process as §9) found
something more consequential than any single bug: **the whole "DriveEngine feeds chat
stance" premise was inert in production.** `chat_stance.py` reads `drive_state` from
substrate snapshots tagged `snapshot_source="drive_state"` — but the only function that ever
produced one, `orion/substrate/adapters/autonomy.py`'s `map_autonomy_artifacts_to_substrate()`,
had zero live callers anywhere in the repo. The function actually registered as the live
"autonomy" producer (`orion/substrate/relational/adapters/autonomy_ctx.py`'s
`map_autonomy_ctx_to_substrate`) hardcodes `snapshot_source="autonomy"` and never calls the
DriveEngine one, despite its own module docstring claiming to. Every test in §9's patch passed
because they all hand-constructed `ctx["chat_drive_state"]`/synthetic snapshots directly,
bypassing the actual broken seam. Net effect: `dominant_drive` in production stayed sourced
from `AutonomyStateV2` the whole time, byte-identical to pre-PR behavior — §9's fixes were
real and correct, just never actually exercised live.

**Fixed by wiring a live producer**, reusing existing infrastructure rather than building a
new one: `services/orion-substrate-runtime/app/worker.py` already ran a
`_drive_state_listener_loop` (previously only caching `DriveStateV1` for an unrelated,
default-off embodiment feature) and already had an established
`store.upsert_node(...)` materialization pattern used by four other live producers in the
same file. Extended that existing subscription — on `DriveStateV1` receipt,
`_materialize_drive_state_to_substrate()` now calls `map_autonomy_artifacts_to_substrate()`
and upserts the resulting nodes into the substrate graph. Added a sibling
`_drive_audit_listener_loop`/`_cache_drive_audit_message` (mirroring the existing
drive-state listener exactly) to also cache the latest `DriveAuditV1`, so
`dominant_drive`/`summary`/`tension_kinds` are available alongside `pressures`/`activations`.
Gated by a new `DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED` (default **on** — this is the
live signal chat stance/Mind depend on, deliberately independent of
`EMBODIMENT_C_TICK_ENABLED`, which gates an unrelated feature and defaults off; the two share
the one bus subscription via `or`, not `and`, so neither silently gates the other).

**AutonomyStateV2 is now fully retired, not demoted.** Per direct instruction: killing it
partially (keep it alive as a fallback, or keep sourcing `active_tensions`/`confidence` from
it) would have repeated the exact "duplicative things serving overlapping purposes" pattern
this whole investigation kept surfacing. The fix that made this possible: `DriveAuditV1`
already had a real `tension_kinds` field (`orion/core/schemas/drives.py`), computed from real
tension events (`orion/spark/concept_induction/audit.py`) — it just was never pulled through
`chat_stance.py`'s projection. Adding that one field closed the only real gap V2 was covering.
`confidence` is dropped entirely, not replaced — V2's version was never real signal (its own
doc: "kind-literal constants (uncalibrated)"). `pressure_trend` stays `None` — no honest
single-turn signal exists for an async-tick system, and this codebase's own convention is to
never fabricate one.

Concretely: `_run_autonomy_reducer` and `_log_autonomy_pressure_probe` deleted outright from
`chat_stance.py` (not flag-gated off — the functions no longer exist), along with their
now-dead imports and the three test files that existed only to exercise them
(`test_chat_stance_autonomy_v2.py`, `test_chat_stance_autonomy_v2_pressure_probe.py`,
`test_chat_stance_autonomy_v2_persistence.py`). `AUTONOMY_STATE_V2_REDUCER_ENABLED` removed
from `.env_example` — it's read nowhere anymore. `autonomy_slice.py` simplified to a single
unified path sourced only from `drive_state` (`dominant_drive`, `active_tensions` from real
`tension_kinds`, `pressure_trend`/`confidence` always `None`) — no more branching, no more
fallback. `orion/self_state/inner_state_registry.py` updated: `autonomy_state_v2`'s status
changed to `REHEARSAL` (`no_cognition_consumer`) with an explicit "RETIRED 2026-07-16" note;
`drive_state.v1`'s `cognition_consumers` now lists its three real live consumers instead of an
empty tuple. The reducer module itself (`orion/autonomy/reducer.py`, `evidence_compiler.py`,
`state_store.py`, `models.py`) is left in place, unused by any live caller — full deletion is
separate, not-yet-done cleanup requiring its own check for stray importers.

Also fixed in this round (from the second review): a missing regression test for the
snapshot cross-field-atomicity fix (§9 shipped the fix but no test could distinguish it from
the pre-fix buggy behavior); a missing regression test that could distinguish "shared pool"
from "duplicate pool" (same issue for the pool-reuse fix); and the new eval file's
import-safety guard, which reimplemented a thinner, weaker copy of an existing guard built
specifically for this repo's three-services-each-ship-a-package-named-`app` collision class
(a class of bug this repo has already hit once, not hypothetical) — now imports the real
guard instead.

## 7. Source material index

- Founding design conversations: external (GPT), not committed anywhere prior to this file —
  the excerpts in §1-2 are the only committed record of them as of 2026-07-16.
- `docs/superpowers/specs/2026-07-07-homeostatic-drives-real-tensions-design.md` — first
  repo-native theory doc; diagnosed and killed the cadence-pinning bug (soft_saturate fixed
  point at ~0.731).
- `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md` — self-initiation
  spec, gated, currently NO-GO.
- `docs/superpowers/specs/2026-07-11-scarcity-economy-brainstorm-v2.md` — reversed the
  internal-economy spec's assumptions against live data (`resource_pressure` was saturated at
  1.0, not dead at 0; since fixed, see `services/orion-spark-introspector/app/worker.py`).
- `docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-audit-design.md` — internal
  audit finding no documented rationale for the six-drive taxonomy or its weights.
- `docs/autonomy_state_v2_reducer.md` / `orion/autonomy/README.md` — operator contract and
  explicit isolation rule for `AutonomyStateV2` vs `DriveEngine`.
- `docs/superpowers/pr-reports/2026-07-08-homeostatic-drives-real-tensions-design-pr.md`
  through `2026-07-14-drive-history-reflection-synthesis-pr.md` — execution history, phased
  gates, honest `DONE_WITH_CONCERNS` reporting throughout.
