# PR: Metric semantic layer, phases 1+2 — URN join + mechanical blast radius

## Summary

- Adds `orion/metrics/` — a **read-only projection** joining the four
  metric-bearing registries into one URN space (587 URNs). Not a sixth
  registry: nothing is hand-authored.
- Adds mechanical downstream-consumer discovery (AST, not regex), answering
  "who else reads this metric" — a question nothing in the repo could answer
  before, including graphify.
- Adds `scripts/check_metric_lineage.py` (summary / `--json` / `--metric` /
  `--drift`) and `make check-metric-lineage`.
- Found and fixed two undercounting bugs in the scanner itself by running it
  against the real repo; both are the exact failure modes CLAUDE.md §0A
  records as incidents.
- Found (not fixed) a stale declared-consumer list pointing at a deleted
  service, and 5 inner-state metrics with no discoverable consumer.
- Liveness verdicts remain **uncomputed** (phase 5) and every card says so
  explicitly, rather than leaving a blank that reads as "fine".

## Outcome moved

The specific failure this targets: *an agent edits a consumer of metric X
without knowing where X comes from, whether it's alive, or who else reads it.*

Before: answering that meant hand grep-archaeology, redone from scratch each
time. The incident record shows it repeatedly not being done at all.

After: `make check-metric-lineage METRIC=prediction_error` returns the
producer, registry of origin, upstream, declared consumers, and 23
mechanically-discovered non-test consumers with file:line and access kind.

Concretely, for `prediction_error` the report lists
`orion/substrate/endogenous_curiosity.py:154` directly beside
`orion/substrate/attention_self_model.py:509` — the generic consumer that the
`transport_prediction_error` retirement missed, next to the named one it
fixed. Retiring a metric by editing one consumer is now *visibly* incomplete.

## Current architecture

Five registries existed, none joined:

| Registry | Granularity | Count |
|---|---|---|
| `orion/bus/channels.yaml` | bus channel | 261 |
| `orion/schemas/registry.py` | schema class | hundreds |
| `orion/inner_state_registry.py` | inner-state signal | 13 |
| `orion/signals/registry.py` | organ | 30 |
| `config/field/field_channel_glossary.v1.yaml` | field channel | 38 |

`orion/field/channel_glossary.py::classify_channel_series()` already provided a
battle-tested liveness classifier — scoped to field channels only — plus the
rule this design preserves: **verdicts are computed, never declared.**

Nothing could resolve a metric across surfaces, and consumer lists were
hand-maintained.

## Architecture touched

Additive only. No service, channel, schema, payload, or env key changed. The
new package reads existing registries and the filesystem; it writes nothing.

## Files changed

- `orion/metrics/__init__.py`: package doc stating the not-a-sixth-registry rule.
- `orion/metrics/lineage.py`: four resolvers + `build_graph()` → `MetricNode` URN space.
- `orion/metrics/consumers.py`: AST scanner, 7 access kinds, blast radius.
- `scripts/check_metric_lineage.py`: CLI.
- `tests/test_metric_lineage.py`: 17 gate tests.
- `Makefile`: `check-metric-lineage` target + `.PHONY`.
- `docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md`: design + build findings.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: purely additive; deleting `orion/metrics/` and the
  script removes the feature with no residue. Nothing is baked into a schema,
  manifest, or training default (§0A step 6, reversibility).

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: not applicable — no env surface touched.
- local `.env` synced: not applicable, no template changed.
- skipped keys requiring operator action: none.

## Why this is not a keyword cathedral

§0A requires a new concept to ship with real mechanism attached. The metric URN
ships with: a producer (four resolvers), a consumer (the CLI + `make` target),
a reducer (`build_graph`), a debug surface (lineage cards), a test suite (17),
and measured runtime output. No URN is hand-authored; a metric absent from
every registry gets no URN, and that absence is reported as the finding.

## Findings (reported, not fixed)

1. **Stale declared-consumer list.** `inner_state_registry.py` declares
   `services.orion-spark-introspector.app.inner_state:build_inner_state_features`
   as the cognition consumer for `self_state.v1` and its scalar fields.
   `services/orion-spark-introspector` **no longer exists** (killed 2026-07-28).
   Surfaced mechanically by `--drift` on first run. Recorded on the agent board.
2. **Malformed producer field.** One registry entry carries free prose in the
   producer position (`orion.spark.concept_induction.drives (DELETED 2026-07-30)`),
   which propagates into its URN. Registry data-quality issue, not a resolver bug.
3. **5 inner-state metrics with no discoverable consumer:** `delta_phi`,
   `delta_recon_error`, `overall_confidence`, `recon_error`,
   `shuffle_baseline_loss`.

All three are contract-surface edits and belong in their own patches.

## Tests run

```text
$ .venv/bin/python -m pytest tests/test_metric_lineage.py -q
17 passed, 2 warnings in 36.97s
```

Covers: all four resolvers non-empty against real registries; import-failure
propagation; URN uniqueness/well-formedness; canonical dimension names excluded
from scan tokens; per-line access-kind classification against a hand-computed
fixture; comment and prose mentions producing no hit; no double-counting;
unparsed files reported not swallowed; `.worktrees/` excluded; test paths
tagged not dropped; plus two live regressions (`prediction_error` →
`endogenous_curiosity.py`, `cpu_pressure` → generic collection consumers).

## Evals run

```text
No eval harness exists for orion/metrics/ (new package).
```

This is observability tooling whose correctness is fully covered by
deterministic gate tests — there is no quality/behavior dimension a periodic
eval would measure that the gate tests do not. Stated rather than claimed as
covered.

## Docker/build/smoke checks

```text
Not applicable -- no service, container, port, dependency, or boot-time
config touched. Runtime evidence is the CLI output below.
```

```text
$ python scripts/check_metric_lineage.py
  bus_channel        261        scan tokens        386
  field_channel       38        files scanned     3714
  inner_state         36        consumer hits     4713
  organ_signal       252        runtime            ~13s
  TOTAL              587 URNs

$ python scripts/check_metric_lineage.py --json | jq length
587

$ python scripts/check_metric_lineage.py --metric not_a_real_metric ; echo $?
UNREGISTERED: 'not_a_real_metric' resolves to no URN in any registry.
1
```

## Review findings fixed

<!-- filled in from the code-review subagent run -->

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity:** low. **Concern:** dynamic metric access
  (`vector[name_from_config]`) is invisible to any static scanner.
  **Mitigation:** undercounts, never overcounts — a reported consumer is always
  real. Stated in the module docstring and the design's Known limits.
- **Severity:** low. **Concern:** the YAML/JSON scan is substring-based and
  noisier than the exact-match Python scan. **Mitigation:** tagged `config` and
  excluded from high-confidence blast radius.
- **Severity:** medium. **Concern:** absence of a liveness verdict could be
  misread as "this metric is fine". **Mitigation:** every lineage card prints
  `Liveness verdict: NOT COMPUTED (phase 5)` plus an explicit warning not to
  read it that way. Phase 5 closes it properly.

## PR link

<https://github.com/junebug-junie/Orion-Sapienform/pull/new/docs/metric-semantic-layer>
