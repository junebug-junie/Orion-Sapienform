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
3. **38 of 57 organ `signal_kinds` in `ORGAN_REGISTRY` have no code consumer
   at all** — including `cognitive_collapse`, `coherence_state`,
   `concept_salience`, `cpu_load`, `disk_io`. This is the STRONG-reliability
   surface (signal kinds are read as dict keys), so the result is meaningful
   rather than a scan artifact. It is the closest thing this patch produces to
   a keyword-cathedral detector pointed at an existing registry.
4. **19 inner-state tokens with no discoverable consumer**, of which the
   scalar-field ones are the strong signal: `delta_phi`, `delta_recon_error`,
   `overall_confidence`, `recon_error`, `shuffle_baseline_loss`,
   `ar1_surrogate_loss`.

All are contract-surface edits and belong in their own patches. None is a
liveness verdict — "no code consumer" and "dead metric" are different claims,
and this patch only supports the first.

## Tests run

```text
$ .venv/bin/python -m pytest tests/test_metric_lineage.py -q
31 passed, 2 warnings in 50.33s
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
  bus_channel        261        scan tokens        391
  field_channel       38        files scanned     3714
  inner_state         37        consumer hits     5938
  organ_signal       252
  TOTAL              588 URNs

  tokens with no discoverable code consumer: 207
  (registry-of-origin excluded; NOT a liveness verdict)
    bus_channel     150   WEAK  -- subscribed via config, not string reads
    inner_state      19   MIXED -- scalar fields strong, *.v1 ids weak
    organ_signal     38   STRONG -- signal_kinds are read as dict keys
    field_channel     0

$ python scripts/check_metric_lineage.py --metric gpu_load | grep BLAST
  BLAST RADIUS (discovered, non-test, high-confidence): 0

$ python scripts/check_metric_lineage.py --json | jq length
588

$ python scripts/check_metric_lineage.py --metric not_a_real_metric ; echo $?
UNREGISTERED: 'not_a_real_metric' resolves to no URN in any registry.
1
```

## Review findings fixed

Code review ran at `high` on `main...docs/metric-semantic-layer` and returned
13 findings. All 13 fixed; every one reproduced by hand before fixing.

A note on process: the first review run reported against the wrong target —
the shared checkout's pre-existing dirty `graphify-out/` tree, not this branch.
Its findings there are real but unrelated to this PR (see **Out-of-scope**
below). It was re-run explicitly scoped to the branch.

- **Finding (HIGH): `iter_source_files` matched excluded dir names against
  absolute path parts.** A scan run from `.worktrees/<name>` or
  `.claude/worktrees/agent-<id>` — both live conventions per CLAUDE.md §2, the
  latter created by the Agent tool's own `isolation: "worktree"` — returned
  **zero files with no error**.
  - Fix: match `path.relative_to(base).parts`.
  - Evidence: reproduced (`iter_source_files` → `[]` with a real consumer file
    present); now gated by a parametrised test over both conventions. The
    original test passed only because `tmp_path` happened to contain no
    excluded component — false confidence, now removed.

- **Finding (HIGH): `_is_float_like`'s PEP-604 branch never fired.**
  `str(types.UnionType)` is `"<class 'types.UnionType'>"`, never
  `"types.UnionType"`, so every `X | None` metric was silently dropped —
  including `FieldStateV1.recent_perturbation_zscore`. `orion/schemas/` uses
  that style predominantly.
  - Fix: `origin is types.UnionType or origin is typing.Union`.
  - Evidence: confirmed on 3.12.3 (`float | None` → `False`); URN count
    587 → 588, with a test asserting the recovered metric by name.

- **Finding (MEDIUM): registry-of-origin files counted as consumers of their
  own metrics.** This inverted the tool's central claim: every organ token got
  a `collection_member` self-hit from `orion/signals/registry.py`, and for
  **38 of 57** that self-hit was the only high-confidence non-test result.
  - Fix: `consumers_for(..., exclude_paths=...)`, fed by
    `graph.registry_sources_for(token)`.
  - Evidence: `gpu_load` reported blast radius **1** before, **0** now — the
    honest answer. Cards print `declared in <file> (excluded from blast radius)`.

- **Finding (MEDIUM): orphan detection was structurally blind to the two
  largest surfaces.** It counted `KIND_CONFIG` hits; every field channel
  appears in the glossary and every bus channel in `channels.yaml`, both inside
  `SCAN_ROOTS` — so 299 of 386 tokens could never be reported orphaned.
  - Fix: orphans judged on real consumers only, broken down per surface.
  - Evidence: reported orphans **5 → 207**. Deliberately *not* reported as
    "207 dead metrics": a string-literal scan cannot judge bus channels
    (subscribed via config), so each surface now prints an explicit
    STRONG/MIXED/WEAK reliability caveat. `field_channel` orphans: **0**.

- **Finding (MEDIUM): name-based dimension blocklist deleted real metrics.**
  It removed the genuine glossary channel `confidence` and 5 real inner-state
  scalars from **both** blast radius and orphan output — invisible in both.
  - Fix: blocklist removed entirely. `scan_token` is structurally always the
    `name` half of a URN, never the `#field` half, so it solved a
    non-existent problem while causing a real one.
  - Evidence: test asserts `confidence` is both a real field channel and a
    live scan token.

- **Finding (MEDIUM): 14 permanently-dangling upstream URNs.** Synthesised
  organ parents (`metric://organ_signal/<p>/<p>`) could never match a real
  node (`/<organ>/<kind>#<dim>`), so causal edges across 252 of 588 nodes
  resolved to nothing.
  - Fix: recorded as `upstream_organs` (organ ids, not URNs).
  - Evidence: new `test_no_dangling_upstream_urns` asserts the set is empty.

- **Finding (MEDIUM): multi-producer bus channel URNs depended on YAML order.**
  41 channels have >1 producer; the rest were dropped silently and a cosmetic
  reorder would rename the URN.
  - Fix: `sorted()`, full set retained on `all_producers`.
  - Evidence: test asserts sorted order and `producer_service ==
    all_producers[0]`.

- **Finding (MEDIUM): `declared_consumers` carried two incompatible meanings**
  — dimension names on the field-channel surface, service names elsewhere —
  printed under one label. A comment also claimed `build_graph()` inverted
  these into `upstream`; it never did.
  - Fix: split out `feeds_dimensions`; false comment removed.

- **Finding (LOW-MED): `visit_Attribute` counted writes as reads.**
  `self.pressure = x` is a write; combined with single-word tokens
  (`pressure`, `phi`, `energy`) this inflated blast radius.
  - Fix: `ast.Load` context only. Evidence: test asserts the write line yields
    no hit.

- **Finding (LOW-MED): `scan_config` substring-matched**, so `cpu_pressure`
  matched `cpu_pressure_ewma`.
  - Fix: whole-token boundary check. This compounded the orphan bug above.

- **Finding (LOW): `ast.parse` raises `ValueError` on embedded NUL bytes**, not
  `SyntaxError` — one such file would abort the entire scan.
  - Fix: caught and routed to `unparsed`. Evidence: test with a real NUL file
    asserts the scan completes and still finds the other consumer.

- **Finding (LOW): `test_registry_import_failure_propagates` asserted on its
  own mock.** It monkeypatched `resolve_inner_state` to a raising stub, then
  asserted the stub raised — it would have passed even if the real code
  swallowed every error.
  - Fix: drives the real resolver via `sys.modules` injection; same gate added
    for `resolve_organ_signals`.

- **Finding (LOW): `--json` silently ignored when `--metric` was given**, while
  the Makefile composes both independently.
  - Fix: emits a JSON lineage card.

Tests: **17 → 31**.

## Out-of-scope issue found (NOT part of this PR)

The mis-targeted first review surfaced a real, live problem in the **shared
checkout**, present before this work started and untouched by it:

`graphify-out/GRAPH_REPORT.md` (tracked, modified) reads **2,471 nodes /
3,210 edges** while its sibling `graphify-out/graph.json` holds **28,306 nodes
/ 81,046 links** — a 91.3% disagreement, the signature of the recurring
graphify destructive-update bug.

Root cause: `scripts/safe_graphify_update.sh` backs up only `$GRAPH_FILE` and
`$MANIFEST_FILE` (L58-59) and restores only those on all three paths (L64-65,
L72-73, L91-92). `graphify update .` also rewrites `GRAPH_REPORT.md`, which is
never captured — so the guard restores the graph, leaves the shrunken report,
and prints "there is nothing to commit," which is false.

This matters because CLAUDE.md directs every agent to read `GRAPH_REPORT.md`
for architecture review. The correct full-size report is already on disk at
`graphify-out/2026-08-12/GRAPH_REPORT.md`. Left for Juniper — it lives in the
shared checkout, which this session does not write to.

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
