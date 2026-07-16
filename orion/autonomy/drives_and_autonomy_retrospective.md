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

- `autonomous_cycle_v1` doesn't exist.
- The one mechanism that could produce true self-origination (`endogenous_origination.py`)
  is code-complete but flag-disabled on a NO-GO whose underlying measurement needs
  re-verification against fresh (non-stale) data.
- The one live consumer of drive state (`chat_stance.py`) is reactive by construction — it
  only runs inside a turn something else already triggered — and reads from the
  turn-scoped, non-decaying `AutonomyStateV2`, not the tested, persistent `DriveEngine`.

The instrument (pressure measurement) is real and well-built. The actuator (unprompted
action) was never built. That is the single most load-bearing fact in this document.

## 6. Open questions

1. Is self-initiated, sometimes-suboptimal behavior still the actual target? If yes, the
   next concrete patch is narrow: wire something to fire without a prompt when `DriveEngine`
   pressure crosses threshold (a journal entry, a probing question) and observe what happens.
2. Re-run the `coactivation_frac` gate against a source certified fresh before treating the
   endogenous-origination NO-GO as still current — the original reading was proven stale once
   already.
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
