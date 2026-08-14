# PR report — definition-change alert over the metric semantic layer (R4)

Branch: `feat/definition-change-alert`

## Summary

- **R4** of `docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md`, the last unbuilt rung. Juniper's ask, verbatim: bus streams and organ signals do not need a liveness verdict, they need "a gate to flag it to me when an agent starts to fuck around in there."
- Diffs the **resolved definition layer** — `orion.metrics.lineage.build_graph()`, 595 URNs across the four registries — against a committed lock, not the YAML text. Cosmetic edits produce zero deltas; one real edit produces one named delta.
- **The alert is the lock diff.** The gate goes red on any definition change and the only way to green is `--update`, which writes the classified deltas into the lock's own `_last_change` block as plain sentences. Re-locking is what writes the sentence, so an agent cannot re-lock quietly.
- Static by construction, so it joins `.github/workflows/orion-static-gates.yml` — unlike the R3 merge-domination gate, which needs live Postgres and had to go to cron.
- Fixes `tests/test_metric_lineage.py`, **red on main since 2026-08-13**, for exactly the reason R4 exists.

## Outcome moved

Three metric renames on 2026-07-24 (`execution_load`, `bus_health`, `transport_pressure`) went unannounced. Three weeks later `execution_load` was still in all four live node vectors, frozen at 0.2672 — a plausible-looking reading with no producer behind it. Found by hand on 2026-08-14.

Replayed that exact rename against this gate:

```
[HIGH] renamed (1)
    renamed metric://field_channel/orion-field-digester/cortex_exec_step_load
         -> metric://field_channel/orion-field-digester/execution_load
[HIGH] routing_changed (1)
    routing_changed metric://bus_channel/orion-cortex-exec/orion:verb:result
         (declared_consumers: ['orion-cortex-orch', 'orion-hub'] -> ['orion-cortex-orch'])
```

And the noise-control claim, verified in the other direction: reordering a consumer list, re-quoting it, and adding a comment to `channels.yaml` → **0 changes, gate PASS**. Removing one consumer from that same list → **HIGH**.

### A live instance, found while building this

`tests/test_metric_lineage.py::test_every_registry_resolves_nonempty` was **red on main**. `fix(bus): retire the drives:audit channel entry` (2026-08-13) took bus channels 261 → 260 for good reasons, and the `>= 261` lower bound was never updated. A deliberate, defensible retirement that surfaced only as an off-by-one nobody read, on main, for a day.

That is R4's failure mode verbatim. The bound is now a floor against rot (`>= 250`) with the counting job handed to the lock, which reports a removal **by name**.

## Current architecture

`orion/metrics/lineage.py` (PR #1603) resolves four registries into one URN space:

| surface | registry | count |
|---|---|---|
| `bus_channel` | `orion/bus/channels.yaml` | 260 |
| `organ_signal` | `orion/signals/registry.py` | 252 |
| `inner_state` | `orion/inner_state_registry.py` | 45 |
| `field_channel` | `config/field/field_channel_glossary.v1.yaml` | 38 |

Nothing consumed that graph as a *change* surface. `check_metric_lineage.py --gate` ratchets orphans; nothing noticed a definition being edited.

## Architecture touched

New leaf module + CLI. No existing caller changed — `grep` for importers of `orion.metrics.definitions` returns exactly the new script and the new test. `git diff main` against `config/field/`, `orion/bus/`, `orion/signals/`, `orion/inner_state_registry.py` is empty: **this branch changes zero metric definitions**, which the gate itself confirms by passing.

## Files changed

- `orion/metrics/definitions.py` (new): fingerprint + classified diff. Pure, no I/O.
- `scripts/check_definition_drift.py` (new): `--gate` / `--update` / `--json`.
- `config/metrics/metric_definitions.lock.json` (new, generated, 247KB): 595 definitions.
- `tests/test_metric_definition_drift.py` (new): 32 tests.
- `.github/workflows/orion-static-gates.yml`: one step.
- `Makefile`: `check-definition-drift` with `GATE=1` / `UPDATE=1`.
- `tests/test_metric_lineage.py`: the stale lower bound above.

## Design decisions, and what each one costs

**Lock, not ratchet.** `orphan_baseline.json` and `merge_domination_baseline.json` may shrink and never grow, because an orphan and a dominated merge are both defects. A definition change is neither — it is an event. So this tracks current truth in both directions and its diff is the deliverable.

**Identity vs content.** `surface/producer/name/field` form the URN, so they cannot "change" — a changed identity is a removal plus an addition. Content splits three ways, and one URN changing in two classes at once emits two changes so a routing edit is never hidden behind a same-commit prose edit:

| class | fields | severity |
|---|---|---|
| semantics | `meaning`, `schema_id` | high |
| routing | `all_producers`, `declared_consumers`, `feeds_dimensions`, `upstream`, `upstream_organs` | high |
| annotation | `notes`, `registry_source` | medium |

Severity ranks **how easy a change is to miss by hand**, not how wrong it is. `added` is medium because a new metric arrives with code that reads it and has its own gate (CLAUDE.md 0A); `removed` is high because nothing arrives at all.

**Rename pairing is deliberately conservative, and here is the number that argues against it.** On the live graph there are 342 distinct fingerprints for 595 nodes — **296 nodes (49.7%) share a fingerprint with at least one other**, almost all `organ_signal`, where 48 nodes collapse to `organ=graph_cognition class=endogenous`.

That is survivable for two measured reasons. The uniqueness guard applies to the removals and additions *in this diff*, not the whole graph, so a 48-node group only goes ambiguous if 2+ members are removed and 2+ added in one change. And `removed` and `renamed` carry the same `high` severity by design, so a mispairing still names both URNs at the same volume — the label degrades, the warning does not. Asserted in `test_thin_definitions_can_mispair_and_that_is_survivable` rather than left as an argument.

A rename that also rewrites the prose — the common case — does **not** pair, and degrades to `removed` + `added`, which is louder. The pairing pass can only ever make output quieter, which is why it is not fuzzy.

## Schema / bus / API changes

None. Read-only over existing registries.

## Env/config changes

None. No env key added, removed, or renamed; `.env_example` untouched, so no sync needed.

## Tests run

```
pytest tests/test_metric_definition_drift.py -q
    32 passed

pytest tests/test_metric_lineage.py tests/test_metric_lineage_gate.py \
       tests/test_metric_definition_drift.py -q
    101 passed in 72.79s
```

Before the lower-bound fix that suite was `1 failed, 100 passed` — the pre-existing main breakage above.

## Evals run

**Mutation testing** — the eval that matters for a diff engine, since every test here asserts a classification and a classification test can pass for the wrong reason. 18 mutants against `orion/metrics/definitions.py`:

```
first run:  KILLED 15/18
  ALIVE: routing -> medium          (no test asserted routing_changed is high)
  ALIVE: field default semantics    (the _FIELD_CLASS fallback was unexercised)
  ALIVE: surface always unknown     (DefinitionChange.surface asserted nowhere)

after adding 4 tests:  KILLED 18/18
```

All three survivors were real gaps, not harness artifacts. The fallback one is the interesting case: it exists so a **stale lock naming a field the current format no longer emits** still diffs instead of raising `KeyError`, and that is now `test_lock_field_retired_from_the_format_degrades_to_annotation`.

Harness at `/tmp/mutate_r4.py`; it restores the source in all paths.

## Docker/build/smoke checks

Not a service change — no container, port, or compose wiring touched. The relevant smoke is the CI dep set, since this joins a workflow that installs exactly three packages:

```
python3 -m venv /tmp/r4venv
/tmp/r4venv/bin/pip install pydantic pydantic-settings PyYAML
/tmp/r4venv/bin/python scripts/check_definition_drift.py --gate
    -> definition drift gate: PASS   (exit 0)
```

Same discipline the workflow header records for the gates already in it: verified in a clean venv, and green on main before being wired in.

```
make check-definition-drift            -> 595 definitions, 0 changed
make check-definition-drift GATE=1     -> PASS, exit 0
make check-definition-drift UPDATE=0   -> lock byte-identical afterwards
```

That last one is not ceremony: `$(if $(VAR),...)` treats **any** value as true, so `UPDATE=0` would have silently rewritten the lock. The Makefile matches explicit true values instead, the same fix `check-metric-lineage-gate` already carries for `UPDATE_BASELINE`.

## Review findings fixed

Code review ran in a subagent; findings and fixes appended below before merge.

## Restart required

```text
No restart required.
```

Read-only static gate. No service reads the lock at runtime.

## Risks / concerns

- **Severity: low.** An agent can still run `--update` and commit without reading the deltas. What it *cannot* do is make them absent from the diff — the mechanism is disclosure, not consent. Stated because it is the honest ceiling on this design.
- **Severity: low.** The lock is 247KB. Git handles it and diffs only show changed entries, but it is the largest generated artifact in `config/`.
- **Severity: low.** `organ_signal` definitions are thin (`organ=X class=Y`), so 49.7% of nodes share a fingerprint — quantified above with why it does not weaken the alert. Enriching the organ registry would improve rename precision and is not in scope here.

## PR link

<pending>
