# Orion Autonomy Phase 3.1 Verification + Observability Harness

## Scope

Phase 3 added autonomy artifact RDF materialization for identity snapshots, drive audits, and proposal-only goals.
Phase 3.1 adds a practical verification harness so developers can replay known artifacts, confirm semantic materialization locally, and optionally verify that the same artifacts are visible through GraphDB.

This task intentionally does **not** introduce execution semantics, autonomous planning, or a new runtime.

## What was added

- `orion/autonomy/verification.py`
  - loads replay fixtures
  - reuses `services/orion-rdf-writer/app/autonomy.py`
  - runs local RDF assertions per artifact class
  - can publish fixture envelopes onto the live bus
  - can run GraphDB ASK checks when GraphDB is configured
  - emits JSON + markdown reports
- CLI entrypoints:
  - `scripts/replay_autonomy_artifacts.py`
  - `scripts/verify_autonomy_graph.py`
  - `scripts/run_autonomy_scenario.py`
- A six-scenario fixture pack in `tests/fixtures/autonomy_graph/scenario_pack.json`
- Regression tests in `tests/test_autonomy_verification_harness.py`

## Scenario coverage

The harness covers the six artifact scenarios requested for practical debugging:

1. self-model snapshot
2. user-model snapshot
3. world-model snapshot
4. relationship-model snapshot
5. drive audit
6. proposal-only goal

## Local verification semantics

The harness verifies all scenarios locally by re-materializing RDF and checking for the following invariants:

- graph URI matches the artifact class
- the artifact node exists via `orion:artifactId`
- `orion:belongsToModelLayer` is present
- `orion:aboutEntity` is present
- provenance exists via `orion:hasProvenance`
- source event references are preserved
- evidence links are preserved
- correlation / trace / turn lineage is preserved
- tension edges are preserved when tension refs exist
- identity snapshots preserve anchor strategy and drive assessments
- world-model snapshots do not collapse to a generic `world` entity
- relationship-model snapshots remain distinct from self-model and user-model entity typing
- drive audits include drive assessments, dominant drive, and active drive highlights
- proposed goals remain `proposal-only` and `proposed`

## GraphDB verification semantics

When `--graphdb` is enabled and `GRAPHDB_URL` / `GRAPHDB_REPO` are configured, the harness runs ASK queries for:

- artifact existence in the expected autonomy graph
- presence of `aboutEntity`
- presence of `belongsToModelLayer`
- presence of an `ArtifactProvenance` node

This is enough to confirm that materialized triples became queryable in the expected GraphDB graph, while keeping the smoke surface small and actionable.

## Readiness assessment

### What is now trustworthy

- The repo now has a repeatable, fixture-driven autonomy verification harness.
- Developers can validate semantic materialization without depending on live infrastructure.
- The same harness can be pointed at live bus + GraphDB infrastructure for operational debugging.
- Reports are generated in both JSON and markdown so the results are easy to archive and compare between runs.

### What remains partial

- End-to-end bus → RDF writer → GraphDB verification is still infrastructure-dependent.
- In this environment, GraphDB-backed verification may be skipped if no reachable GraphDB instance is configured.
- The harness verifies publishability and stored graph shape, but it does not add a new explicit RDF-writer ingestion acknowledgment beyond existing platform behavior.

## Commands

### Replay fixture artifacts locally

```bash
python scripts/replay_autonomy_artifacts.py \
  --dry-run \
  --scenario world-model-snapshot \
  --scenario proposal-only-goal \
  --json-out tmp/autonomy_replay_report.json \
  --md-out tmp/autonomy_replay_report.md
```

### Replay fixtures onto the bus

```bash
python scripts/replay_autonomy_artifacts.py \
  --publish-bus \
  --scenario drive-audit \
  --wait-sec 3 \
  --json-out tmp/autonomy_replay_report.json \
  --md-out tmp/autonomy_replay_report.md
```

### Verify graph semantics locally

```bash
python scripts/verify_autonomy_graph.py \
  --json-out tmp/autonomy_verification_report.json \
  --md-out tmp/autonomy_verification_report.md
```

### Verify local + GraphDB semantics

```bash
python scripts/verify_autonomy_graph.py \
  --graphdb \
  --scenario self-model-snapshot \
  --scenario drive-audit \
  --json-out tmp/autonomy_verification_report.json \
  --md-out tmp/autonomy_verification_report.md
```

### Run the combined scenario harness

```bash
python scripts/run_autonomy_scenario.py \
  --scenario self-model-snapshot \
  --scenario drive-audit \
  --scenario proposal-only-goal \
  --json-out tmp/autonomy_scenario_report.json \
  --md-out tmp/autonomy_scenario_report.md
```

### Run the combined scenario harness with live dependencies

```bash
python scripts/run_autonomy_scenario.py \
  --scenario world-model-snapshot \
  --scenario drive-audit \
  --scenario proposal-only-goal \
  --publish-bus \
  --graphdb \
  --wait-sec 3 \
  --json-out tmp/autonomy_scenario_report.json \
  --md-out tmp/autonomy_scenario_report.md
```

## Bottom line

**Yes:** the repo now has a practical autonomy observability / testing harness.

**Partially:** graph materialization is verified end-to-end only when the live bus and GraphDB path are reachable; otherwise local semantic verification is still strong, but operational confirmation is partial.

**Main remaining blocker:** infrastructure-level observability for the live ingestion path, especially reliable confirmation that a published autonomy artifact was consumed and persisted downstream in the expected graph.
