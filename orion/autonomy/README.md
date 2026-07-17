## Origin and current gap

Why any of this exists, the founding theory, and the biggest unresolved gap (self-initiation
was never built): [drives_and_autonomy_retrospective.md](drives_and_autonomy_retrospective.md)

## Hub Drives Analytics

Design: [docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md](../../docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md).

**Why this surface exists.** The six-drive `DriveEngine` economy (`orion/spark/concept_induction/drives.py`,
`DRIVE_KEYS = ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")`) has run
live for a while — per-tick audits, `tick_attribution`/`tension_kinds` on the wire, Postgres history, offline
gate scripts — but until this patch, seeing any of it meant CLI archaeology
(`scripts/analysis/measure_autonomy_gate.py`, `scripts/drive_state_divergence_audit.py`) or a per-turn chat
side panel. The Hub **Drives** tab (`#drives`, standalone page `/drives-analytics`) is an orientation and
observability surface: gauges for current pressure per drive, real contributor history pulled from
`tick_attribution` (not invented), a program-health KPI strip, dual time series (tick-rate + six pressure
sparklines), goal-alignment coloring, and a divergence/integrity check against the `drive_state.v1` concept
store. It **does not mutate** drives, tensions, or goals — there is no promote/dismiss/complete/adjust control
anywhere on this tab, by design (see the spec's non-goals).

**What "good" looks like on this tab.** Healthy is *churn* — pressures rising and falling across the six
drives as tensions resolve — combined with goals actually forming out of elevated pressure (a drive's
pressure is high **and** an active goal's `drive_origin` matches it: goal-aligned "green" in `align`/`combined`
color mode). It is explicitly **not** "all six pressures pinned high." Six-way saturation/monoculture is a
regime the tab colors **red**, the same as a stale audit rail or a starved drive — high magnitude alone never
reads as healthy. `build_window_kpis`/`drive_economy_verdict_from_drive_stats`
(`services/orion-hub/scripts/drives_analytics.py`) reuse the saturation/coactivation thresholds from
`scripts/analysis/measure_autonomy_gate.py` so the Hub strip's `gate_verdict_drive_only` agrees with the
offline gate's math; it can report `GO_DRIVE_ONLY` (drive-rail coactivation cleared, not saturated) but never
claims a full offline-gate `GO`, since that also requires `resource_pressure`, which this drive-only endpoint
does not have.

**Where to look next.**

- Full origin story and the still-open self-initiation gap: [drives_and_autonomy_retrospective.md](drives_and_autonomy_retrospective.md)
- Operator tab mechanics (iframe isolation, endpoints, hash params, restart order): [services/orion-hub/README.md](../../services/orion-hub/README.md)
- The tab itself: Hub shell hash `#drives`, standalone page `/drives-analytics`
- Deeper offline measurement (GO/NO-GO/SATURATED/UNMEASURABLE including `resource_pressure`): `scripts/analysis/measure_autonomy_gate.py`

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

The module below is left in place, unused by any live caller — full deletion is separate,
not-yet-done cleanup.

| Piece | Path | Role (historical) |
|-------|------|------|
| Schema | `orion/autonomy/models.py` | `AutonomyEvidenceRefV1` optional `signal_kind` / `dimension` / `value` |
| Compiler | `orion/autonomy/evidence_compiler.py` | Omit-when-empty gates from stance locals (not `ctx["chat_social_bridge_summary"]`) |
| Adapter | `orion/autonomy/signal_tension.py` | `chat_evidence_to_tension` — direct map lookup, no DeviationGate/EWMA |
| Map | `config/autonomy/signal_drive_map.yaml` | `chat_social_hazard` + `chat_reasoning_quality` rows |
| Reducer | `orion/autonomy/reducer.py` | Fold `magnitude * drive_impacts` into `drive_pressures`; return `tensions_minted` |

Operator contract (historical): [docs/autonomy_state_v2_reducer.md](../../docs/autonomy_state_v2_reducer.md)

The module's own tests still pass (it's dead code, not broken code) if you need to verify it in isolation:
```bash
PYTHONPATH=. python orion/autonomy/evals/run_autonomy_v2_movement_eval.py

pytest orion/autonomy/tests/test_evidence_compiler.py \
  orion/autonomy/tests/test_signal_tension.py \
  orion/autonomy/tests/test_autonomy_reducer.py \
  orion/autonomy/tests/test_autonomy_isolation.py -q
```

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
