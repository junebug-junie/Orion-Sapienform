## The six drives — conceptual audit (2026-07-11)

`DRIVE_KEYS = (coherence, continuity, capability, relational, predictive, autonomy)` (`orion/spark/concept_induction/drives.py`) has no conceptual grounding document anywhere in this repo — no docstring, no design note arguing for this set over any other. It was introduced once, early, and every later spec treats it as fixed ("Six drives stay" — hard constraint in `docs/superpowers/specs/2026-07-07-homeostatic-drives-real-tensions-design.md`). The 2026-07-07 leaky-integrator fix made the *pressure math* honest (no more flat-0.731 pin); it did not make the *taxonomy* honest.

**Not concept-induction-derived.** Despite living in `orion/spark/concept_induction/`, drives never reference `ConceptProfile`/`ConceptProfileDelta` (verified by grep, zero hits). Naming-proximity artifact only.

**Two unsynchronized computations share the six keys:**
- `orion.spark.concept_induction.drives.DriveEngine` — leaky integrator, feeds `GoalProposalEngine`.
- `orion.autonomy.reducer.reduce_autonomy_state` — RDF-persisted, feeds `AttentionItemV1` generation + `capability_policy` drive-origin gating (this file's "AutonomyStateV2 evidence" section above).

Same six names, same `config/autonomy/signal_drive_map.yaml`, different math, different store, not read from each other.

**Operational semantics** (what actually fires each one, since no docstring states an intended meaning):

| Drive | Actual triggers |
|---|---|
| `coherence` | drop in self-state/turn `coherence` score, `spark_signal.coherence` dip |
| `continuity` | novelty spikes, uncertainty deltas, biometric volatility, mesh-health drops |
| `capability` | energy/resource/execution pressure, biometric strain, failure severity |
| `relational` | valence drops, social-hazard signals (cooldown loops, self-message loops) |
| `predictive` | coherence deltas, uncertainty, novelty, world-coverage gaps |
| `autonomy` | novelty, uncertainty, low feedback scores |

`coherence`, `continuity`, and `predictive` draw from largely the same underlying tensions (self-state coherence/uncertainty deltas, novelty) with different weight vectors — three views on one signal, not obviously three distinct constructs. `autonomy` is named for a capacity for self-initiation but its inputs are the same generic distress signals as three other drives; the one mechanism that would actually earn the name — `orion/autonomy/endogenous_origination.py` — is real but `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false` by default. The spec names a known gap (no self-preservation drive; biometric somatic signals get mapped onto `capability`/`continuity` as a workaround) and explicitly declines to solve it.

None of this is theater in the empty-shell sense — every drive has a real, live producer. The open question is whether these are the right six categories, or six labels stamped onto overlapping slices of one tension stream. Full writeup: `docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-audit-design.md` (gitignored, local only — ask Juniper if it needs to travel with a specific branch).

## Subject routing

Autonomy goals and drives are keyed by subject (`orion`, `relationship`, `juniper`). Dyadic chat materializes to **relationship**, not juniper — see the routing contract:

- [Autonomy subject routing contract](../../docs/architecture/autonomy_subjects.md)

## AutonomyStateV2 evidence (chat, env-gated)

Optional turn-local reducer that upgrades graph `AutonomyStateV1` with **typed** evidence and map-driven pressure math. Keyword substring pressure is **removed**.

| Piece | Path | Role |
|-------|------|------|
| Schema | `orion/autonomy/models.py` | `AutonomyEvidenceRefV1` optional `signal_kind` / `dimension` / `value` |
| Compiler | `orion/autonomy/evidence_compiler.py` | Omit-when-empty gates from stance locals (not `ctx["chat_social_bridge_summary"]`) |
| Adapter | `orion/autonomy/signal_tension.py` | `chat_evidence_to_tension` — direct map lookup, no DeviationGate/EWMA |
| Map | `config/autonomy/signal_drive_map.yaml` | `chat_social_hazard` + `chat_reasoning_quality` rows |
| Reducer | `orion/autonomy/reducer.py` | Fold `magnitude * drive_impacts` into `drive_pressures`; return `tensions_minted` |

**Hard isolation:** this path must not wire into phi, `build_self_state`, or homeostatic `DriveEngine`.

Operator contract + enable bar: [docs/autonomy_state_v2_reducer.md](../../docs/autonomy_state_v2_reducer.md)

```bash
# Enable bar (must exit 0 before flipping the flag in cortex-exec)
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
