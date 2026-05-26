## Subject routing

Autonomy goals and drives are keyed by subject (`orion`, `relationship`, `juniper`). Dyadic chat materializes to **relationship**, not juniper — see the routing contract:

- [Autonomy subject routing contract](../../docs/architecture/autonomy_subjects.md)

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
