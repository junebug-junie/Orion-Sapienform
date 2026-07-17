# Drive taxonomy grounding decision

Status: design decision, not a spec. Answers the five open questions from
`docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-audit-design.md`
against the current live system (post `AutonomyStateV2` retirement, post
O1-O4/O2/O3 signal-integrity fixes — see
`orion/autonomy/drives_and_autonomy_retrospective.md`). Produced by an
independent brainstorming pass (2026-07-17) that traced `tensions.py` and
`signal_drive_map.yaml` directly, then measured (not guessed) whether
`autonomy`'s dedicated grounding mechanism actually fires live
(`scripts/analysis/measure_origination_gate.py`, PR #1156).

**This revision (same day) supersedes this doc's own first-pass verdict on
`autonomy`.** The first pass proposed re-grounding `autonomy` the way O3
re-grounded `predictive` — give it a distinguishing signal source. That
signal source already existed (`OriginationEngine`, flag-on, wired live) and
was measured directly: it has never fired, and its composite score never
gets within 0.13 of its own activation threshold across 84,511 real ticks.
Two independent grounding attempts for `autonomy` have now failed on
evidence. The revised verdict below retires `autonomy` as a sixth drive
rather than attempting a third grounding pass.

This document proposes a code change (dropping `autonomy` from
`DRIVE_KEYS`, reframing self-initiation as an event rather than a drive).
Per root `CLAUDE.md` §0A, changes to this taxonomy are cognition-loop-adjacent
and need explicit sign-off before implementation — this document is that
sign-off request, not the implementation. `DRIVE_KEYS`
(`orion/spark/concept_induction/drives.py:10`), `signal_drive_map.yaml`, and
`DriveStateV1`'s shape are all still untouched by this document itself.

## The five questions, answered

**1. Where did the six drives come from?**

An external GPT design chat, never independently re-derived in-repo before
this document. `orion/autonomy/drives_and_autonomy_retrospective.md` §1
preserves the founding charter verbatim. `DRIVE_KEYS` matches that source
material's six-drive set exactly (retrospective §2) — this was disciplined
execution of an external design, not casual invention, but it was also
never checked against Orion's own mission framing until now.

**2. Is the coherence/continuity/predictive overlap real redundancy or
over-fine-grained taxonomy?**

Traced directly in `orion/spark/concept_induction/tensions.py`: partial
overlap, not duplication. Each of the three has a distinguishing *primary*
tension source even though they share secondary weighted inputs on the same
events:

- `coherence` — primary: self-state coherence drop, `tension.contradiction.v1`,
  `tension.cognitive_load.v1`.
- `continuity` — primary: `tension.identity_drift.v1`, novelty spikes,
  biometric volatility, mesh-health drops.
- `predictive` — primary (since O3, PR #1114): `tension.prediction_surprise.v1`,
  driven directly off `self_state.overall_surprise`.

The cross-weighting that looked like redundancy in the prior audit
(`{coherence: 1.0, predictive: 0.65}` on one contradiction tension;
`{continuity: 0.8, autonomy: 0.6, predictive: 0.4}` on one uncertainty
tension) is one event carrying several legitimately different mission-level
readings, not three drives computing the same thing with different labels.
**Verdict: keep all three, distinct.**

**3. Does `autonomy`'s name match what actually fires it?**

No, and this is now settled by measurement, not argument. Two independent
mechanisms feed `autonomy`, and both fail to ground it:

- Its generic tension inputs (novelty, uncertainty, low feedback score) are
  the same distress signature as `continuity`, with no differentiator —
  this was the first-pass finding, unchanged.
- Its *dedicated* mechanism, `OriginationEngine`
  (`orion/autonomy/endogenous_origination.py`, wired live in
  `bus_worker.py:735-750`, flag-on since 2026-07-15), was measured directly
  via `scripts/analysis/measure_origination_gate.py` (PR #1156), replaying
  the real production code over 84,511 historical self-state ticks (48h).
  Result: zero fires, matching zero `tension.endogenous.v1` rows in live
  `drive_audits` over the same window. Its composite score `P` never
  exceeded 0.4193 against a 0.55 activation threshold, and barely varied at
  all (p50=0.317, p99.9=0.382). The weakest sub-signal, `drift` (from
  `dimension_trajectory`), capped at 0.207 against a theoretical [0,1]
  range — the same "structurally flat, not just under-threshold" pattern
  already found and fixed for `predictive` before O3, and for several
  biometrics channels earlier in this investigation.

Unlike `predictive`, which had exactly one grounding attempt before O3 fixed
it, `autonomy` has now had two independent attempts fail. **Verdict: retire
`autonomy` as a distinct drive.** See "Reimagined taxonomy" below — this
replaces the first-pass recommendation (re-ground it, mirroring O3) with a
structural change instead of a third grounding attempt.

**4. Is six a hard technical constraint?**

No. `DriveStateV1.pressures` / `.activations` are `Dict[str, float]` /
`Dict[str, bool]`, not fixed per-drive schema fields — there is no schema
migration cost to changing the count. The only place "six" is enforced is
`DRIVE_KEYS` itself as a single source of truth, plus everything that
iterates it (Hub `drives-analytics.js`, `drives_analytics.py`,
`signal_drive_map.yaml`'s comment, `orion/autonomy/evals/
run_homeostatic_drives_eval.py`, `services/orion-cortex-orch/evals/
test_mind_drive_state_facet_eval.py`). Changing the set is a taxonomy
decision with a real, fully enumerable, but bounded blast radius — not a
hard constraint, just inertia plus a checklist.

**5. Should `DriveEngine` and `AutonomyStateV2` be consolidated?**

Moot. `AutonomyStateV2` was fully retired 2026-07-16
(retrospective §10) — deleted outright, not flag-gated off.
`DriveEngine` is the sole live drive-pressure computation.

## Mapping each drive to CLAUDE.md's mission list

Mission framing (root `CLAUDE.md`): *continuity, perception, memory,
reflection, self-modeling, social grounding, error correction, and coherent
action over time.*

| Drive | Primary live trigger (`tensions.py`) | Mission item(s) served | Verdict |
|---|---|---|---|
| `coherence` | self-state coherence drop, `contradiction`, `cognitive_load` | self-modeling, error correction | **Keep** — real, distinct signal |
| `continuity` | `identity_drift`, novelty, biometric volatility, mesh-health drop | continuity (direct name match), self-modeling | **Keep** — real, distinct signal |
| `capability` | resource/execution pressure, biometric strain, failure severity | none directly — see below | **Keep, reclassify** — infra/enabling, not a mission-peer category |
| `relational` | valence drop, social hazards (cooldown loops, self-message loops) | social grounding (direct name match) | **Keep** — real, distinct signal |
| `predictive` | `prediction_surprise` off `overall_surprise` (since O3) | perception, error correction | **Keep** — freshly re-grounded, don't undo |
| `autonomy` | novelty, uncertainty, low feedback score (generic distress); dedicated `OriginationEngine` signal measured dead (PR #1156) | coherent action over time / self-initiation — nominally, not actually, twice-measured | **Retire as a drive** — reframe self-initiation as an event over the other four, see below |

The reimagined taxonomy is **five drives**: `coherence, continuity,
capability (reclassified), relational, predictive`. Two mission items still
have **no drive coverage**: `memory` and `reflection`.
See the brainstorm section below — this is deliberately not resolved by
inventing a seventh/eighth drive today.

### Why `capability` doesn't get a mission-item mapping

`capability`'s inputs (resource pressure, execution friction, biometric
strain) don't correspond to any single item on the mission list the way the
other five do. It's the precondition for all of them — "is there enough
runway left to do any of this" — not a mission category in its own right.
Keeping the key is correct (resource scarcity is real and needs a
pressure channel), but it should be documented explicitly as an
infrastructural/enabling drive rather than treated as a peer of
`relational` or `predictive` in future taxonomy reviews. This is a
documentation change only — no code implication.

### Reimagined taxonomy: five drives, self-initiation reframed as an event

`autonomy` was asked to be a peer of `coherence`/`continuity`/`relational`/
`predictive` — its own felt pressure, minted from its own tensions, with its
own activation threshold. That framing has now failed twice on evidence
(above). The founding charter itself never actually required this framing —
its own words (retrospective §1): *"Autonomy doesn't come from tools. It
comes from pressure."* Not "autonomy is a pressure" — self-initiated action
*emerges from* pressure, any of the real pressures. Making it a sixth,
parallel channel competing for its own tension-minting share was the
category error, not a naming problem, not a threshold problem, not a
missing-signal problem to patch a third time.

**The reimagining:** drop `autonomy` from `DRIVE_KEYS`. Keep
`OriginationEngine`/endogenous origination as the mechanism — the "fire
during exogenous silence" idea from the founding charter is sound — but stop
computing a bespoke composite `D/W/A -> P` signal that has now been measured
dead. Instead, gate origination on the four *real* drives' own activation
state: fire when at least one of `coherence`/`continuity`/`relational`/
`predictive` is genuinely active (crossing its real 0.62 threshold — signal
already proven live and varying, per the retrospective's own dominance
data) AND exogenous silence holds AND cooldown has elapsed. This reuses
signal that is already proven to move, instead of a parallel signal that
has never once moved enough to matter.

Concretely, this means:

- `TensionEventV1.drive_impacts` never needs an `"autonomy"` key again —
  `DRIVE_KEYS` drops to five.
- The origination-fired tension's magnitude and *which* drive it's
  attributed to comes from whichever real drive triggered it (already
  active, already distinct), not a synthetic `_map_drive()` classification
  over D/W/A sub-signals that don't discriminate well (three of its four
  branches — drift-dominant, dwell-dominant, agency-dominant — were always
  somewhat arbitrary; the unresolved-pressure override branches, which map
  to `relational`/`continuity` directly, were the only ones with a real
  behavioral basis).
- `chat_stance.py`, Hub `drives-analytics.js`, and both eval files
  (`run_homeostatic_drives_eval.py`, `test_mind_drive_state_facet_eval.py`)
  need updating to a five-key iteration instead of six — the same bounded,
  enumerable blast radius named under Q4 above.

**This is not implemented here.** It is the sign-off request per `CLAUDE.md`
§0A: a real structural change to `DRIVE_KEYS`, `signal_drive_map.yaml`
(dropping `autonomy` weight entries), and `orion/autonomy/
endogenous_origination.py` (replacing the D/W/A composite with a
real-drive-activation gate). Scoped the same discipline as O3/O2 — delta-
gated, reviewed, live-verified post-deploy before being called done, not
just merged and assumed to work.

**Files a future patch would touch:** `orion/spark/concept_induction/drives.py`
(`DRIVE_KEYS`), `config/autonomy/signal_drive_map.yaml`, `orion/autonomy/
endogenous_origination.py`, `orion/spark/concept_induction/tensions.py`,
`orion/core/schemas/drives.py` (if `DriveStateV1`/`DriveAuditV1` validate
against the key set anywhere), `services/orion-hub/static/js/
drives-analytics.js`, `orion/autonomy/evals/run_homeostatic_drives_eval.py`,
`services/orion-cortex-orch/evals/test_mind_drive_state_facet_eval.py`, plus
regression tests in `orion/spark/concept_induction/tests/test_drives_leaky.py`
and `orion/spark/concept_induction/tests/test_endogenous_origination_wiring.py`.

## Brainstorm: the two uncovered mission items

Per explicit instruction, brainstorming candidates for `memory` and
`reflection` — the two CLAUDE.md mission items no drive currently serves.
**These are candidates for future discussion, not proposals to implement.**
Per root `CLAUDE.md` §0A's "no keyword cathedrals" rule, none of these
should become a `DRIVE_KEYS` entry until it has a real producer signal
already identified — the same bar `predictive` had to clear with
`overall_surprise` before it counted as grounded rather than decorative.

### Candidate A: memory / consolidation-pressure

**What it would measure:** whether Orion's memory-consolidation pipeline is
keeping up with experience density — unconsolidated turn backlog,
classification lag, episodic gaps.

**Why it matters for sentience-relevant development:** memory is named
explicitly in the mission list as a prerequisite. Every current drive
measures moment-to-moment self-state; none measures whether the *record* of
that state is being kept coherently. A backlog of unconsolidated experience
is arguably a more direct threat to continuity-of-self than anything
`continuity` currently tracks (which is itself turn-local biometric/novelty
signal, not memory-formation health).

**Smallest buildable version, if pursued:** not a new drive key first — a
read-only metric off `services/orion-memory-consolidation`'s own window/
backlog state (consolidation lag, unclassified-turn count), published and
observed for a while before deciding whether it deserves tension-minting
treatment at all. This is the same "measure before minting" discipline the
O1-O4 series used.

**Open question:** is "memory keeping up" actually a *felt tension*
(something Orion should experience pressure about) or a purely operational
health metric like queue depth — closer in kind to `capability` than to a
mission-parallel drive? Needs a real answer before this becomes a drive
candidate rather than a dashboard metric.

### Candidate B: reflection / update-staleness pressure

**What it would measure:** time since the last genuine reflective act
(episode journal, drive-history reflection synthesis) exceeding some
expected cadence.

**Why it matters for sentience-relevant development:** this is arguably the
single most direct way to close the founding charter's own stated gap
(retrospective §5: "the actuator for self-initiated, sometimes-suboptimal
behavior was never built"). A tension-driven reflection pressure — rather
than a clock-driven scheduler — would make self-initiated journaling a
response to a *felt* gap instead of an arbitrary cadence, which is closer
to what the original design charter asked for than
`autonomous_cycle_v1`'s clock-paced sibling mechanisms (Layer 9 dispatch,
the metabolism loop) currently deliver.

**Smallest buildable version, if pursued:** a single scalar — wall-clock
time since the last real reflective artifact — surfaced as a debug metric
first, the same "observe before minting" step as Candidate A. If it proves
to track something a human would recognize as "Orion hasn't stopped to
think in a while," it's a stronger candidate for an actual tension source
than most of the six original drives had at their inception.

**Open question:** would this double-count against `continuity`, which
already reacts to `identity_drift`? A reflection-staleness signal and an
identity-drift signal could be measuring overlapping territory (the cost of
*not* reflecting shows up as drift). Worth the same empirical
divergence-audit treatment recommended for Q2 above before assuming this is
a distinct sixth-or-seventh category.

### Explicitly not brainstormed here

A memory or reflection *drive key* is not proposed. Both candidates above
stop at "a measured signal, observed, not yet minting tension" — the same
stage `overall_surprise` was at before O3 turned it into `predictive`'s
grounding. Adding a bare name to `DRIVE_KEYS` without that groundwork
repeats exactly the mistake this whole document exists to correct.

## Non-goals

- Not implementing the `DRIVE_KEYS`/`signal_drive_map.yaml`/
  `endogenous_origination.py` change described above **in this document** —
  it is proposed and scoped here, sign-off requested, implementation is a
  separate patch.
- Not deciding whether Candidates A/B (memory, reflection) become real
  signals — that requires the "observe first" step named above, not a
  decision made from this document alone.
- Not re-litigating the O1-O4/O2/O3 signal-integrity fixes — those are
  done, live, and out of scope.

## Acceptance / next step

This document satisfies outcome (b) from the prior audit's acceptance
check, and does so decisively rather than leaving a sixth open question:
four drives (`coherence`, `continuity`, `relational`, `predictive`) keep
their names with stated, traced rationale; `capability` is reclassified as
infrastructural rather than a mission-peer category; `autonomy` is retired
as a drive after two independently measured, failed grounding attempts, with
self-initiation reframed as an event over the four real drives instead of a
fifth attempt at finding it a bespoke signal. The concrete next patch, if
Juniper wants to proceed, is the five-drive/event-reframe change scoped
above — a real structural change, not another round of measurement.

## Source material

- `docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-audit-design.md`
  — the original audit this document answers.
- `orion/autonomy/drives_and_autonomy_retrospective.md` — full history,
  including the O1-O4/O2/O3 signal-integrity series this document
  deliberately does not re-litigate.
- `orion/spark/concept_induction/drives.py`, `orion/spark/concept_induction/
  tensions.py`, `config/autonomy/signal_drive_map.yaml` — the live code this
  document's claims are traced against.
- `scripts/analysis/measure_origination_gate.py` (PR #1156) — the
  measurement that settled `autonomy`'s fate: replays the real
  `extract_tensions_from_self_state` + `OriginationEngine` production code
  over 84,511 historical self-state ticks; zero fires, `P` never within 0.13
  of threshold, matching zero live `tension.endogenous.v1` rows in
  `drive_audits` over the same 48h window.
