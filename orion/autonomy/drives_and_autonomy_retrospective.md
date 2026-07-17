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

### 5b. Status update (2026-07-17): O2 and O3 shipped, live-verified, and a full trace from a
new fold-batch saturation mechanism all the way back to a pre-existing, one-day-old open
question in `orion-field-digester`'s own channel glossary

This is the long version, written so nobody has to re-derive any of it from scratch. Short
version first, detail after.

**Short version.** O3 (`predictive` re-grounding) and O2 (event-rate normalization) — §5a's two
named follow-ups — both shipped and merged same-day. Live post-deploy verification showed the
dominance-attribution fix (O1) and the starved-drive fix (O3) both genuinely working in
production (top dominant-drive share dropped from 96% to 31.65%; `predictive` went from ~0% to a
real 7.8% presence) — but the gate still read SATURATED, and a live agent-board finding showed
why: O2's fold-batching design has its own saturation failure mode, distinct from the one it
fixed, which a subsequent multi-hop investigation traced all the way upstream through the
self-state substrate, through `orion-biometrics`' publish cadence, and into a **confirmed root
cause in `orion-field-digester`'s decay/injection-interval mismatch** — the exact mechanism
behind an "accumulator-oscillation artifact" that service's own channel glossary had already
flagged as unconfirmed one day earlier (2026-07-16). Nothing here has been patched yet except O2
and O3 themselves; the field-digester fix and the DriveEngine fold-batch fix are both still open,
both cognition/substrate-adjacent infrastructure changes that need explicit sign-off before
anyone touches them.

**O3 shipped** (PR #1114, merged 2026-07-17T04:31:47Z, branch
`feat/drive-predictive-reground-o3`). `predictive` had never had a primary tension source — only
secondary `drive_impacts` weights (0.4–0.7) riding other tensions (`tension.contradiction.v1`,
`tension.identity_drift.v1`) — and sat dead at a 0.016 median pressure. Fix: a new block in
`extract_tensions_from_self_state()` (`orion/spark/concept_induction/tensions.py`) minting
`tension.prediction_surprise.v1` off `self_state.overall_surprise`, a real, already-clamped
top-level field on `SelfStateV1` that this function had simply never read. The first version of
this patch fired unconditionally whenever `overall_surprise > 0.30`, on every tick — a full
`/code-review medium` pass caught that this had no delta-gating (unlike its four sibling blocks
in the same function) and proved, with real numbers, that it would pin `predictive`'s pressure
above 0.95 within ~5 ticks (~23 seconds) at bus cadence: the exact saturation pattern this patch
existed to fix, reintroduced for a different drive. Fixed by switching to the same delta-gated
pattern (`surprise_delta > 0.05` once a previous self-state exists; absolute-threshold fallback
only on the very first tick) every sibling block already used. This is the first of several times
this same lesson recurs in this write-up: **a decay mechanism whose injection frequency isn't
gated against its own decay rate will eventually saturate, and every fix in this whole
investigation is a variation on that one failure mode.**

**O2 shipped** (PR #1126, merged 2026-07-17T08:07:34Z, branch `feat/drive-cadence-fold-o2`).
`DriveEngine.update()` was called once per raw bus event (~13/min, ~4.6s apart); with
`decay_tau_sec=1800.0`, decay between consecutive calls is ~0.997 (negligible), so repeated
same-direction impulses converge pressure toward 1.0 within seconds regardless of tau — the
event-rate/decay mismatch §5a named as the still-open half of the diagnosis, and the same
mechanism behind the cpu/gpu_pressure saturation bugs fixed in PRs #1108-1111. Fix: a new
`ConceptWorker._update_drive_pressures()` (`orion/spark/concept_induction/bus_worker.py`) splits
read from write — every bus event still gets a fresh decay-only pressure projection
(`tensions=[]`, so goal proposal / dossier / identity snapshot / publish all keep getting a live
`drive_state` every event, unchanged), but new tension impulses are buffered per-subject in
memory and only actually folded into the persisted integrator at most once per 900 seconds
(`_DRIVE_FOLD_INTERVAL_SEC`, a plain module constant, not an env key — first-pass calibration,
`p* = impulse/(1-decay*(1-impulse))` worked out to ≈0.63 at a representative 0.4 impulse, right
at the 0.62 activation threshold). `DriveEngine` itself is completely untouched — this is purely
a call-cadence fix in the stateful caller. An 8-angle review found and fixed a real unbounded
pending-tension buffer (capped at 500, drop-oldest — matching this repo's established
`evidence_event_ids <= 200` convention) and a duplicate `DriveEngine.update()` call in the
fold/non-fold branches (collapsed to one); it also traced and *cleared* three other candidates as
accepted, documented tradeoffs rather than bugs: goal-proposal priority scoring's pressure term
can lag a freshly-dominant drive by up to 900s (accepted — `dominant_drive` itself still reacts
immediately), `scripts/drive_state_divergence_audit.py`'s `AutonomyStateV2` comparison side is
already frozen/historical (retired 2026-07-16) so a less-frequently-updated `DriveEngine` side
introduces no new noise, and the in-memory pending buffer is lost on worker restart with the same
risk profile every other per-subject buffer in this file already carries (`self.window`,
`self.last_run`, `self.recent_event_seen`). The PR report explicitly flagged, as an **unverified
risk**: *"if the folded impulse itself gets clamped near 1.0 (many co-firing tensions summed
within the 900s window before the `_clamp_signed([-1,1])` ceiling in `DriveEngine.update()`),
decay stops mattering and pressure still approaches 1.0 regardless of interval choice... cannot
be ruled out without live data."* This risk materialized within hours of deploy — see below.

**Live post-deploy verification** (2026-07-17, ~08:27 UTC, ~20 minutes after O2 deployed).
`scripts/analysis/measure_autonomy_gate.py --window-hours 6` needs `POSTGRES_URI` overridden to
`postgresql://postgres:postgres@localhost:55432/conjourney` when run from the host — its default
DSN uses the Docker-internal hostname `orion-athena-sql-db`, unresolvable outside the compose
network. Results: dominance attribution is genuinely fixed live, not just in tests — top
dominant-drive share dropped from the 96% relational monoculture §5a documented down to **31.65%**,
and `predictive` went from ~0% presence to **356 audits (7.8%)**, real and nonzero for the first
time. But gate verdict (b) still read **SATURATED**: `all_active_frac` (drives at pressure ≥0.62)
was 95.9% — most drives still elevated. A close look at the raw time series in the ~20 minutes
immediately after restart showed genuinely smooth decay in progress (`predictive` 0.951→0.90,
`relational` 0.988→0.91) — proof the decay-only-projection mechanism was working correctly — but
this reflected the *legacy*, pre-fix saturated baseline still unwinding from a JSON store file
that survives worker restarts; the 6-hour gate window was still ~95% pre-deploy data. The
original plan at this point was to wait 4-6 hours for a genuinely clean post-deploy window before
re-measuring. That plan turned out to be moot — see below.

**The fold-batch collapse regression** (found live by a separate agent, posted to the shared
agent board, independently verified here). Live container log evidence:

```text
08:39:40  pressures={'coherence': 0.5726, 'capability': 0.4951, 'predictive': 0.5883, ...}
          folded=False, pending_buffered=209
08:39:44  pressures={'coherence': 1.0,    'capability': 1.0,    'predictive': 1.0,    ...}
          folded=True,  tensions_folded=209
```

Real, differentiated per-drive pressure going in; all six drives pinned to the *exact same*
ceiling value coming out of a single fold. Mechanism: `DriveEngine.update()`'s `impact_sum[drive]`
(`orion/spark/concept_induction/drives.py:70-80`) accumulates *unbounded* across every tension in
a fold batch — only the final sum is clamped to `[-1, 1]` via `_clamp_signed`
(`drives.py:91`). Once that clamp hits exactly `1.0`, `pressures[drive] = clamp01(base +
1.0*(1-base)) = 1.0` **regardless of `base`** — the prior differentiated pressure is completely
erased. With ~200 tensions accumulating per 900s fold window (verified: ~0.2-0.25/sec sustained,
matching the raw self-state tick rate) and most tension-minting blocks in this codebase touching
multiple drives per tension, a fold batch this large plausibly saturates every drive's
`impact_sum` simultaneously — explaining why *all six* collapsed together, not just one dominant
drive, and then decayed together in lockstep afterward (still exactly equal to each other). This
is, in one sense, *worse* than the pre-O2 state: before, different drives had different
saturation levels (relational pinned near 0.98, predictive dead at 0.016); the fold-collapse
regression homogenizes all six into one indistinguishable value every ~900 seconds.

Working through the math independently confirmed this is real, and also confirmed something
non-obvious: **a shorter fold interval alone does not reliably fix it.** Total exponential decay
over any wall-clock window is the same regardless of batching granularity — `exp(-900/1800) ≈
0.607` for one 900-second chunk is essentially identical to `exp(-4.6/1800)^195 ≈ 0.602` for 195
tiny ~4.6-second chunks, since exponential decay composes multiplicatively independent of step
size. If roughly 200 same-direction tensions arrive within about one decay half-life-ish window
regardless of how you batch them, *any* reasonable aggregation strategy — summed-then-clamped
(today's design) or applied sequentially with diminishing per-tension headroom (`p += impulse *
(1-p)`, repeated) — converges close to the ceiling; sequential application would at least avoid
the "all six land on the *exact same* value" symptom (200 iterations of that recurrence still
asymptotes near 1.0, just without the hard collapse-to-identical-ceiling that the current
sum-then-clamp design produces), but it does not solve saturation on its own. **The real lever is
upstream: how many tensions get *minted* in a given window, not how the integrator aggregates
them once they exist.** This repo already has a primitive built for exactly that job
(`orion.autonomy.tension_ratelimit.TensionRateLimiter`, a sliding-window drop-over-cap limiter),
just not wired into the main `handle_envelope` tension-minting path.

A genuinely useful side discussion (still exploratory, not decided, raised while reasoning
through the above): a log-odds/logit-space accumulation with exponential forgetting (a
Beta/logistic filter — evidence summed in an unbounded logit space, squashed through a sigmoid
only at the very end) would structurally avoid the "many drives collapse to the identical
ceiling" symptom specifically, since different total evidence amounts always map to numerically
distinguishable sigmoid outputs even near saturation, and it would give a natural
confidence/variance signal alongside the point estimate almost for free. It does **not** by
itself answer whether ~200 tensions/900s is legitimate accumulated evidence or an artifact of
redundant re-minting from one upstream source (see below — turns out to be the latter, at least
in large part), and it is a meaningfully bigger lift than anything in this series so far:
`DriveEngine`'s activation thresholds (0.62/0.42) and the SATURATED gate's own calibration
constants are all tuned against the current linear `[0, 1]` pressure scale and would all need
re-deriving against a logit scale. Not started. Not decided. Named here so it doesn't have to be
re-derived if it comes up again.

**"Look for realness first" — tracing whether the tension volume is genuine evidence or an
artifact, hop by hop.** Live query against `drive_audits.tension_kinds` over a trailing hour:
`tension.distress.v1` 392/hr (by far the largest contributor), `tension.drive_competition.v1`
160/hr (signal-only since O1, contributes zero pressure by design), `tension.prediction_surprise.v1`
135/hr, `tension.cognitive_load.v1` 62/hr, `tension.identity_drift.v1` 56/hr,
`tension.contradiction.v1` 56/hr, `tension.signal.v1` 26/hr — roughly 887/hr total, matching the
raw self-state tick rate closely enough that essentially every tick mints at least one tension.

Querying `substrate_self_state` (Postgres) directly over a 20-minute window surfaced the real
finding: `agency_readiness` is not drifting, it is executing a **mechanical, near-perfect
sawtooth** — a smooth ramp from ~0.28 up to ~0.95 over roughly 15-16 ticks (~30-32 seconds), then
a hard snap straight back down to ~0.28, repeating continuously, for as long as the window was
sampled. `social_pressure` is not near-zero organically — it is **numerically dead**, decaying by
a constant ~92% ratio every tick from some ancient seed value down through `1e-156`/`7e-157`
territory (the same "decayed to numerical dust, functionally inert" pattern this project already
found once before, for `predictive`, ahead of O3).

Since `agency_readiness_score = coherence − 0.25·execution_pressure − 0.35·reliability_pressure −
0.25·uncertainty − 0.15·resource_pressure` (`orion/self_state/scoring.py`), the sawtooth traces to
its inputs: `coherence`, `uncertainty`, and `resource_pressure` are locked in a **tighter,
synchronized sawtooth with an almost-exact ~16-second period** (7-8 ticks) — `coherence` rises
smoothly 0.47→0.84 then snaps back on schedule every cycle, while `uncertainty` and
`resource_pressure` mirror it inversely in perfect sync (expected for `uncertainty`, since
`uncertainty_score = overall_salience * (1 - coherence)` is directly derived from `coherence`).
`reliability_pressure` is completely dead (subnormal float `3e-323`). `execution_pressure` is
smoothly, monotonically decaying with no fresh input at all across the sampled window. The
`tension.distress.v1` fire condition (`combined = -(agency_delta) + social_delta) / 2`, fires when
`combined > 0.04`) fires with large magnitude on exactly the tick `agency_readiness` snaps down —
this is the concrete mechanism behind that tension kind's 392/hr dominance.

Tracing further upstream: `coherence_score()` (`orion/self_state/scoring.py`) subtracts a penalty
over four raw channel-pressure keys — `failure_pressure`, `execution_friction`, `staleness`, and a
generic `pressure` key. That generic `pressure` channel (which also directly feeds
`resource_pressure`'s `channel_dimension_map` entry) is a **diffused aggregate of
`cpu_pressure`/`gpu_pressure`/`memory_pressure`/`disk_pressure`/`thermal_pressure`** per the
topology edges in `config/field/orion_field_topology.v1.yaml`. `config/self_state/self_state_policy.v1.yaml`'s
own comments document that this diffused channel has a known history of "sticking saturated" — a
live incident dated 2026-07-10, with a dedicated evidence-only bypass built specifically to work
around it (`orion-spark-introspector` reads the raw hardware-channel names directly rather than
the diffused `pressure` value, precisely to avoid this).

Tracing into `orion-biometrics`: `TELEMETRY_INTERVAL=30` (the actual host-metrics re-measurement
cadence) and `CLUSTER_PUBLISH_INTERVAL=15` (a re-broadcast of a weighted average of the latest
received summaries) — matching the observed ~30s and ~16s periods closely enough to be the
obvious first suspect. `BiometricsHub.publish_cluster()` was read in full and is **not** itself
injecting anything artificial: it's a straightforward weighted-average re-broadcast of whatever
its `_latest_summary` cache holds, nothing more. `services/orion-biometrics/app/metrics.py` (raw
`/proc/stat`/`/proc/meminfo`/`nvidia-smi` collection via `collect_gpu_stats`) and
`orion/telemetry/biometrics_pipeline.py`'s `_summarize()` (`cpu_pressure = clamp01(max(cpu_util,
load1/cores))`, `mem_pressure = clamp01((mem_total-mem_avail)/mem_total)`, etc.) were both read in
full — both are straightforward, correct, *instantaneous*-fraction calculations. No ratchet, no
unbounded accumulator, no periodic reset anywhere in either file. This is where the trace nearly
stopped, on the (reasonable but incorrect) hypothesis that the sawtooth was genuine, if mysterious,
periodic host telemetry rather than a code defect.

**The confirmed root cause: `orion-field-digester`'s decay/injection-interval mismatch.**
`orion-field-digester` (`services/orion-field-digester/`) is the actual consumer that turns
biometrics readings into `FieldStateV1` node/capability vectors, on a `perturb → decay → diffuse →
suppress` pipeline (`run_digestion_tick`, `app/tensor/update_rules.py`) ticking every
`RECEIPT_POLL_INTERVAL_SEC=2.0` seconds. Two mechanisms, both real, working against each other:

- `apply_decay()` (`app/digestion/decay.py`) multiplies every channel in `NODE_DECAY_CHANNELS`
  (`cpu_pressure`, `memory_pressure`, `gpu_pressure`, `thermal_pressure`, `disk_pressure`,
  `staleness`, `failure_pressure`, `execution_friction`, `reliability_pressure`, and more) by
  `BIOMETRICS_FIELD_DECAY_RATE=0.92` **every single 2-second tick, unconditionally** — whether or
  not fresh biometrics data arrived that tick.
- `apply_perturbations()` (`app/digestion/perturbation.py`) applies a fresh biometrics reading via
  `mode="replace"`: `node_vec[channel] = max(0.0, min(1.0, p.intensity))` — a **full overwrite**,
  not a blend, whenever new data lands (every ~15-30s, per `orion-biometrics`' own publish
  cadence).

The math: `0.92^7 ≈ 0.56` — a channel loses ~44% of its value across the ~15-second gap between
publishes (7-8 ticks at 2s each), then gets snapped straight back up to whatever the current real
measurement is on the next publish, with zero memory of the decayed trajectory in between. This
produces a mechanical sawtooth **regardless of whether the real underlying host metric is stable
or bursty** — it is an artifact of decay-every-tick-unconditionally plus reset-via-full-replace,
not a reflection of genuine host volatility.

This is the exact same bug *class* as O2's fold-batch collapse and O3's original absolute-threshold
firing: a decay mechanism whose injection cadence isn't reconciled against its own decay rate.
The parameter regime is the mirror image, though — `DriveEngine`'s decay was too *weak* relative
to its injection frequency (pinning upward); `orion-field-digester`'s decay is *strong enough*
relative to its injection interval to produce a dramatic, visible swing (a sawtooth) rather than
pinning flat. This almost certainly **predates today's work entirely** — `RECEIPT_POLL_INTERVAL_SEC=2.0`
and `BIOMETRICS_FIELD_DECAY_RATE=0.92` are long-standing configured defaults
(`services/orion-field-digester/.env_example`), and this exact service's own README — in a "Field
channel glossary" section written 2026-07-16, one day before this investigation — already flagged
`cpu_pressure`/`gpu_pressure` as *"real signal — continuous, but flagged as a known
accumulator-oscillation artifact (~60s beat); not confirmed whether that beat reflects real
hardware load or a polling-architecture artifact."* This investigation resolves that open
question: it is the polling-architecture artifact, specifically the decay/injection-interval
mismatch above, not real hardware load — see
[`services/orion-field-digester/README.md`](../../services/orion-field-digester/README.md)'s
updated channel-glossary entries for the full mechanism written up in place.

Blast radius is wider than the drive economy: `NODE_DECAY_CHANNELS` feeds every consumer of
`FieldStateV1` node/capability vectors, not just self-state's `coherence`/`agency_readiness`/
`resource_pressure` — attention scoring and capability-vector consumers read the same channels and
would show the same artifact.

**Where this leaves things (2026-07-17, end of this investigation).** Two real, confirmed bugs are
sitting open, unpatched, both requiring explicit sign-off before anyone touches them (both are
cognition/substrate-adjacent infrastructure, not a narrow, low-risk seam):

1. `DriveEngine.update()`'s fold-batch clamp collapse (`drives.py:70-91`) — all six drives can
   snap to an identical ceiling value on a large enough fold batch. Fixing this alone, without
   addressing item 2, would likely just move the saturation symptom around rather than resolve it
   (per the "shorter interval doesn't fix it" math above) — the tension *volume* is the deeper
   problem, not solely how the integrator aggregates a given volume.
2. `orion-field-digester`'s `apply_decay`/`apply_perturbations` mismatch (`decay.py`,
   `perturbation.py`) — the mechanical source of a meaningful fraction of that tension volume
   (via `tension.distress.v1`'s dependence on `agency_readiness`'s sawtooth), and a
   longer-standing bug than anything else in this whole series.

The originally-planned "wait 4-6 hours, re-run the clean-window gate measurement" plan (end of
the previous status update above) is now understood to be premature: the gate will very likely
keep reading SATURATED regardless of elapsed time until at least one of these two issues is
addressed, since neither is a matter of the legacy baseline simply needing more time to decay.
The Bayesian/log-odds redesign for `DriveEngine` (discussed above) remains a live, undecided
design option, not yet worth committing to until it's clear how much of the volume problem item 2
actually resolves on its own.

Two open, unrelated-but-adjacent agent-board items worth cross-referencing if picking this up:
`pressure_organ.py`'s untested blast-radius expansion from the 3 new memory/thermal/disk hint keys
(PR #1111), and a live field-channel-glossary/`bus_observer` investigation thread that overlaps
this exact territory (`orion-field-digester`'s channel corpus) but was not the source used to
reach the conclusions above.

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
5. Open, confirmed, unpatched (see §5b): `DriveEngine.update()`'s fold-batch clamp collapse can
   snap all six drives to an identical ceiling value on a large enough batch of same-tick
   tensions. Needs a decision on whether to rate-limit tension *minting* upstream (this repo
   already has `TensionRateLimiter` for exactly this, unwired from this path), redesign
   `DriveEngine`'s aggregation math (the log-odds/logit-space option sketched in §5b), or both.
6. Open, confirmed, unpatched (see §5b): `orion-field-digester`'s `apply_decay` (unconditional
   per-tick multiplicative decay) vs. `apply_perturbations`' `mode="replace"` full-overwrite
   produces a mechanical sawtooth on every biometrics-sourced field channel, independent of real
   host telemetry. This predates the O1-O4/O2/O3 series and has a wider blast radius (attention
   scoring, capability vectors) than just the drive economy. Resolves a previously-unconfirmed
   question in `services/orion-field-digester/README.md`'s own channel glossary
   (`cpu_pressure`/`gpu_pressure`'s "accumulator-oscillation artifact, not confirmed... hardware
   or polling-architecture", written 2026-07-16).

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
- `docs/superpowers/pr-reports/2026-07-16-drive-economy-desaturation-o1-o4-pr.md` — O1+O4,
  the dominance-attribution/SATURATED-verdict fix that started the diagnosis §5b completes.
- `docs/superpowers/pr-reports/2026-07-17-drive-predictive-reground-o3-pr.md` — O3 (PR #1114),
  including the delta-gating fix a full code review caught before merge.
- `docs/superpowers/pr-reports/2026-07-17-drive-cadence-fold-o2-pr.md` — O2 (PR #1126),
  including the explicitly-flagged-as-unverified fold-batch-clamp risk that materialized live
  hours later (§5b).
- `services/orion-field-digester/README.md`'s "Field channel glossary" section — the
  2026-07-16 channel-by-channel audit that first flagged the `cpu_pressure`/`gpu_pressure`
  "accumulator-oscillation artifact" as unconfirmed; §5b resolves it.
