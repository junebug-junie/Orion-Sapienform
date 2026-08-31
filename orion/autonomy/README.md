## Program status (2026-08-31): the budget ENFORCES, and Orion made a choice

`orion/autonomy/allocator.py` and the motor-seconds budget stopped being observers.
Until 2026-08-30 both ran in shadow mode -- `motor_allocator_preview` logged what
*would* have been refused and the dispatcher then sent everything anyway. As of
PR #2002 they enforce, and as of PR #2004 there is something outward for them to
choose between.

**The first real choice, live:** 11 candidates in one tick, 1 admitted, 10 refused,
53 motor-seconds spent on making an image -- with the visual chain's 600 s cron
switched **off**. The action ran because it won on value-per-motor-second, not
because a timer fired.

### Two ordering flaws that had been defeating this machinery

Both are worth remembering, because both made the allocator look like it was working
while it was being bypassed:

1. **A priority pre-filter ran before the value scorer.** `max_dispatch_candidates: 5`
   truncated the candidate list by hand-authored `base_priority` *before* `allocate()`
   saw it -- so a constant was choosing what the value machinery was allowed to
   consider. Raised to 50 (above the 11/frame observed live). The real send caps
   (`max_dispatches_per_tick`, the budget, the allocator) are untouched.

2. **The information floor is an absorbing state for expensive actions.** A cold
   posterior yields a fixed `0.5*ln(1+sigma^2/tau^2)` ~ 0.99 nats. Divided by cost,
   **anything costing more than ~49 s can never clear a 0.02 nats/sec floor** -- and so
   can never accumulate the observations that would let it clear one. The floor was
   refusing an action that was 71x better per second than the best alternative
   (`0.018517` vs `0.000259`) and reporting "nothing worth doing".

   `Candidate.cold_start` now exempts **zero-observation** candidates from the
   information floor, and only from that floor: the harm gate, the cost requirement,
   the ordering, and the daily allowance all still apply, and the exemption ends at the
   first observation. **The floor was not lowered** -- its job is retiring things we
   have *learned* are uninformative, which it cannot do to something never measured.

### Diagnostic lesson (cost hours, twice in one day)

`refusals={'unmeasurable': 2}` -- an aggregate that says *how many* but never *which* --
led to a confidently wrong diagnosis. One DEBUG line per candidate settled it in
seconds. The same shape had just cost hours on `substrate_mutation_*` starvation
(PR #1999). **An aggregate that cannot distinguish causes will hide one indefinitely.**

### Known defect, unfixed

`services/orion-thought/app/visual_chain.py`'s single-flight lock can wedge: observed
returning `already_in_flight` while circe was idle, cleared only by a container restart.
A severed HTTP request appears to leave it held, which silently makes the only outward
action permanently unschedulable, with no error. Highest-value open fix here.

Full accounts: `docs/superpowers/pr-reports/2026-08-30-motor-budget-enforcement-pr.md`,
`docs/superpowers/pr-reports/2026-08-31-express-outward-action-pr.md`, and the roadmap
at `docs/superpowers/specs/2026-08-30-self-calibration-roadmap-and-session-handoff.md`.

## Program status (2026-07-30 update): drives-system DELETED, not just halted

The 2026-07-18 halt described below has been followed through: `orion.spark.concept_induction.
drives.DriveEngine` and its whole call chain (tension extraction, `GoalProposalEngine`,
drive-audit publishing) were deleted outright 2026-07-30, along with this module's
homeostatic deviation-tension source (`signal_drive_map.py`, `signal_tension.py`,
`tension_ratelimit.py`, `deviation_gate.py` — see the "AutonomyStateV2 evidence" section
below for a caveat on same-named-but-unrelated prior deletions). Orion lost live
goal-proposal capability from this path as a direct, accepted consequence; no field-native
replacement exists yet. `orion.autonomy.endogenous_origination` was already removed earlier,
independently (2026-07-22). The rest of this file below describes the pre-deletion halted
state; read it for history, not as a description of what currently runs.

## Program status (2026-07-18, historical): drives-system development halted

This module's drive/origination system (`orion.spark.concept_induction.drives.DriveEngine`,
`orion.autonomy.endogenous_origination`) is superseded by
[`orion/sentience_striving_program/README.md`](../sentience_striving_program/README.md),
which governs Orion's motivational/attention/capability-gating substrate going forward.
Read that charter's §8 before further work here — `endogenous_origination.py`'s composite
D/W/A signal was measured (PR #1156) to have never fired across its deployed lifetime;
`capability_policy.py`'s static per-cycle budget is slated for replacement by live,
already-existing field-attention salience (Objective 2), not further drive-taxonomy tuning.

## `ActionOutcomeRefV1.surprise` is not a real signal (found 2026-07-24)

Despite the name, `surprise` (`models.py`) is a binary success/fail proxy at every real call
site (`0.0 if success else 1.0`), never a continuous epistemic-uncertainty measure — and live
`action_outcomes` data confirms it reads exactly `0.0` for every row observed so far. Do not
reuse it as an Active-Inference/epistemic-value term. See the field's own docstring and
`docs/superpowers/specs/2026-07-24-efe-capability-gate-design.md` for the design this was
ruled out of and what real signal replaces it instead.

## Origin and current gap

Why any of this exists, the founding theory, and the biggest unresolved gap (self-initiation
was never built): [drives_and_autonomy_retrospective.md](drives_and_autonomy_retrospective.md)

## Drive-economy desaturation series — O2/O3 shipped, two new bugs found live (2026-07-17)

O3 (`predictive` re-grounding, PR #1114) and O2 (event-rate normalization, PR #1126) — the two
named follow-ups from the 2026-07-16 desaturation diagnosis — both shipped and merged. Live
post-deploy verification confirmed the dominance-attribution fix (O1) and the starved-`predictive`
fix (O3) are genuinely working in production: top dominant-drive share dropped from a 96%
`relational` monoculture to 31.65%, and `predictive` went from ~0% presence to a real 7.8%.

The gate then read SATURATED for a *different* reason than before, and a full multi-hop
investigation traced it all the way back to a **confirmed root cause in `orion-field-digester`**
— not a new bug this series introduced, but a longer-standing one it surfaced. Two real,
confirmed bugs were found; one is now fixed:

1. **Fixed and live-verified 2026-07-17.** `DriveEngine.update()`'s fold-batch clamp collapse —
   a large enough batch of same-tick tensions could snap multiple drives to an identical value,
   erasing real differentiation — was confirmed live (`coherence`/`capability`/`predictive`
   byte-identical at `0.45036942460343243` across consecutive `drive_audits` rows) and fixed by
   switching `update()`'s live path from sum-then-clamp-once to a sequential per-tension update
   (`docs/superpowers/specs/2026-07-17-drive-engine-fold-batch-clamp-collapse-fix-design.md`,
   PR #1148). Post-deploy verification confirmed **zero** collapse events (any drive pair
   identical) since the fix went live (`2026-07-17 19:23:48 UTC`). The specific tie observed
   before the fix shipped resolved on its own via a real fold ~10.5h after forming — see §5e for
   the corrected trace (an earlier version of this note claimed differentiated tensions were
   never reaching the fold buffer for this trio; that was wrong, drawn from too short an
   observation window).
2. **Fixed 2026-07-17.** `orion-field-digester`'s `apply_decay`/`apply_perturbations` mismatch —
   an unconditional per-tick decay fighting a full-overwrite-on-fresh-data reset produced a
   mechanical sawtooth on biometrics-sourced field channels, independent of real host telemetry.
   This resolved a previously-unconfirmed question in that service's own README (see
   [services/orion-field-digester/README.md](../../services/orion-field-digester/README.md)'s
   channel glossary, `cpu_pressure`/`gpu_pressure` entries) and was fixed the same day: channels
   now hold flat until genuinely stale instead of decaying unconditionally every tick
   (`docs/superpowers/specs/2026-07-17-field-digester-decay-hold-fix-design.md`).

Full trace, live evidence, exact mechanisms, and the still-open design questions (rate-limit
tension minting upstream vs. redesign `DriveEngine`'s aggregation math, a sketched
log-odds/logit-space alternative):
[drives_and_autonomy_retrospective.md §5b](drives_and_autonomy_retrospective.md#5b-status-update-2026-07-17-o2-and-o3-shipped-live-verified-and-a-full-trace-from-a-new-fold-batch-saturation-mechanism-all-the-way-back-to-a-pre-existing-one-day-old-open-question-in-orion-field-digesters-own-channel-glossary)

## Hub Drives Analytics — REMOVED 2026-08-13

This section described the Hub `Drives` tab (`#drives`, standalone `/drives-analytics` page),
an orientation/observability surface over the six-drive `DriveEngine` economy's Postgres
history. The tab, its backend, and the underlying `drive_audits` table it read from have all
been removed outright — see `services/orion-hub/README.md`'s "5.4 Drives Analytics panel"
entry and `docs/superpowers/pr-reports/2026-08-13-remove-hub-drives-analytics-tab-pr.md` for
the removal. `DriveEngine` itself was already retired 2026-07-30 (drive-pressure/goal-generation
deletion sprint, `orion/sentience_striving_program/README.md` sec8); this tab had been kept
alive afterward as a deliberate "kill the producer, not the reader" historical-forensics view,
then removed once that history itself was no longer worth keeping around.

Design spec (historical, not updated): [docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md](../../docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md).
Origin story and the still-open self-initiation gap: [drives_and_autonomy_retrospective.md](drives_and_autonomy_retrospective.md).

## Subject routing

Autonomy goals and drives are keyed by subject (`orion`, `relationship`, `juniper`). Dyadic chat materializes to **relationship**, not juniper — see the routing contract:

- [Autonomy subject routing contract](../../docs/architecture/autonomy_subjects.md)

## AutonomyStateV2 evidence — RETIRED 2026-07-16

~~Optional turn-local reducer that upgrades graph `AutonomyStateV1` with **typed** evidence and map-driven pressure math.~~
**Retired, not demoted.** `chat_stance.py`'s call site (`_run_autonomy_reducer`) was deleted
outright — not flag-gated off. `AUTONOMY_STATE_V2_REDUCER_ENABLED` no longer exists anywhere.
`DriveEngine`'s `drive_state` (including its real `tension_kinds`, pulled through as of this
round — see the retrospective §10) is the sole live drive/tension signal for chat stance and
the `orion-cortex-orch`-triggered Mind path now, with no fallback. See
[drives_and_autonomy_retrospective.md §10](drives_and_autonomy_retrospective.md#10-second-round-fix-the-wiring-was-dead-in-production-and-v2-is-now-fully-retired-2026-07-16)
for the full story, including why the wiring in the first round of this fix never actually
activated in production.

The table below is a historical record of a now-fully-deleted module, kept for archaeology
only — do not treat any path in it as present on disk. `evidence_compiler.py`,
`reducer.py`, `run_autonomy_v2_movement_eval.py`, `test_evidence_compiler.py`, and
`test_autonomy_reducer.py` were already gone before 2026-07-30. `signal_tension.py` and
`config/autonomy/signal_drive_map.yaml` were deleted 2026-07-30 as part of a separate,
later sprint (drive-pressure/goal-generation deletion, `orion/sentience_striving_program/
README.md` §8) — same filenames, unrelated deletion event, do not conflate the two.

| Piece | Path | Role (historical) |
|-------|------|------|
| Schema | `orion/autonomy/models.py` | `AutonomyEvidenceRefV1` optional `signal_kind` / `dimension` / `value` |
| Compiler | `orion/autonomy/evidence_compiler.py` (deleted) | Omit-when-empty gates from stance locals (not `ctx["chat_social_bridge_summary"]`) |
| Adapter | `orion/autonomy/signal_tension.py` (deleted 2026-07-30) | `chat_evidence_to_tension` — direct map lookup, no DeviationGate/EWMA |
| Map | `config/autonomy/signal_drive_map.yaml` (deleted 2026-07-30) | `chat_social_hazard` + `chat_reasoning_quality` rows |
| Reducer | `orion/autonomy/reducer.py` (deleted) | Fold `magnitude * drive_impacts` into `drive_pressures`; return `tensions_minted` |

Operator contract (historical): [docs/autonomy_state_v2_reducer.md](../../docs/autonomy_state_v2_reducer.md)

The command block that used to verify this module in isolation is no longer runnable —
every file it referenced except `test_autonomy_isolation.py` is deleted. Do not re-add it.

## Chat stance drives (Hub compact card)

On `chat_stance`, Orion’s drives graph is large and often exceeds SPARQL budgets. Defaults:

| Variable | Default | Effect |
|----------|---------|--------|
| `AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES` | `true` | Skip Orion `drives` subquery; use relationship drives + Orion goals |
| `AUTONOMY_CHAT_STANCE_DRIVES_QUERY_LIMIT` | `20` | Row cap when defer is off |
| `AUTONOMY_DRIVES_SUBQUERY_TIMEOUT_SEC` | `12` | Drives-only timeout when defer is off |
| `AUTONOMY_CHAT_STANCE_SUBQUERY_MAX_WORKERS` | `1` | Serialize identity/drives/goals per subject under load |

Set `AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES=false` only if Orion drives must appear on every chat turn and Fuseki keeps p95 under the drives timeout.

## Goal graph hygiene (automated — do not run host scripts)

| Mechanism | Service | When |
|-----------|---------|------|
| Backlog drain | `orion-actions` | First scheduler tick after deploy (`ACTIONS_DAILY_GOAL_ARCHIVE_RUN_ON_STARTUP=true`) |
| Nightly maintenance | `orion-actions` | 03:15 local (`ACTIONS_DAILY_GOAL_ARCHIVE_*`) |
| Post-publish trim | `orion-spark-concept-induction` | After goal materialization (`AUTONOMY_GOAL_ARCHIVE_ENABLED=true`) |

`scripts/autonomy/archive_stale_goal_proposals.py` is for operator dry-run/debug only. Production path is container automation with Fuseki URLs from each service `.env`.

Run tests:

Prove local semantics:
```
export PYTHONPATH=$PWD
python -m scripts.verify_autonomy_graph \
  --json-out tmp/autonomy_verification_report.json \
  --md-out tmp/autonomy_verification_report.md

cat tmp/autonomy_verification_report.md
```

Prove combined scenario locally:
```
python -m scripts.run_autonomy_scenario \
  --scenario self-model-snapshot \
  --scenario drive-audit \
  --scenario proposal-only-goal \
  --json-out tmp/autonomy_scenario_report.json \
  --md-out tmp/autonomy_scenario_report.md
```

Prove live path:
```
python -m scripts.run_autonomy_scenario \
  --scenario world-model-snapshot \
  --scenario drive-audit \
  --scenario proposal-only-goal \
  --publish-bus \
  --graphdb \
  --wait-sec 3 \
  --json-out tmp/autonomy_scenario_report.json \
  --md-out tmp/autonomy_scenario_report.md
```
