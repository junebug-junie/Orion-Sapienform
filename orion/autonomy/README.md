## Subject routing

Autonomy goals and drives are keyed by subject (`orion`, `relationship`, `juniper`). Dyadic chat materializes to **relationship**, not juniper — see the routing contract:

- [Autonomy subject routing contract](../../docs/architecture/autonomy_subjects.md)

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
