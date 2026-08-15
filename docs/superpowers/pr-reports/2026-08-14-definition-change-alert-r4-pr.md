# PR report — definition-change alert over the metric semantic layer (R4)

Branch: `feat/definition-change-alert`

## Summary

- **R4** of `docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md`, the last unbuilt rung. Juniper's ask, verbatim: bus streams and organ signals do not need a liveness verdict, they need "a gate to flag it to me when an agent starts to fuck around in there."
- Diffs the **resolved definition layer** — `orion.metrics.lineage.build_graph()`, 595 URNs across the four registries — against a committed lock, not the YAML text. Cosmetic edits produce zero deltas; one real edit produces one named delta.
- **The alert is the lock diff.** The gate goes red on any definition change and the only way to green is `--update`, which writes the classified deltas into the lock's own `_last_change` block as plain sentences — derived from the **merge base**, so repeated re-locks cannot overwrite an earlier delta. `--gate` then recomputes that block and fails if the committed one disagrees, which makes the sentence a constraint rather than a convention.
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
- `tests/test_metric_definition_drift.py` (new): 45 tests.
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

**Rename pairing, and the number that set its strictness.** On the live graph there are 342 distinct fingerprints for 595 nodes — **296 nodes (49.7%) share a fingerprint with at least one other**, almost all `organ_signal`, where 48 nodes collapse to `organ=graph_cognition class=endogenous`.

My first version treated a mispair as cosmetic, on the grounds that `removed` and `renamed` are both `high`. Review showed that reasoning is wrong: "renamed A → B" tells the reader the metric still exists under a new name and no consumer needs migrating, which is the *inverse* of the truth when A was deleted — and a deleted metric still being read is the R4 motivating incident itself. Severity survives a mispair; the reader's next action does not.

So uniqueness is required **across the whole lock**, not just within the diff. A definition must identify exactly one metric before and one after. A real rename of a field channel (distinctive `meaning`) still pairs; two interchangeable organ dimensions never do. Both directions are asserted.

A rename that also rewrites the prose — the common case — does **not** pair, and degrades to `removed` + `added`, which is louder. The pairing pass can only ever make output quieter, which is why it is not fuzzy.

## Schema / bus / API changes

None. Read-only over existing registries.

## Env/config changes

None. No env key added, removed, or renamed; `.env_example` untouched, so no sync needed.

## Tests run

```
pytest tests/test_metric_definition_drift.py -q
    45 passed

pytest tests/test_metric_lineage.py tests/test_metric_lineage_gate.py \
       tests/test_metric_definition_drift.py -q
    114 passed in 85.33s
```

Before the lower-bound fix that suite was `1 failed` — the pre-existing main breakage above.

## Evals run

**Mutation testing** — the eval that matters for a diff engine, since every test asserts a classification and a classification test can pass for the wrong reason. Run twice, before and after the review fixes.

Round 1, 18 mutants against `orion/metrics/definitions.py`:

```
KILLED 15/18
  ALIVE: routing -> medium          (no test asserted routing_changed is high)
  ALIVE: field default semantics    (the _FIELD_CLASS fallback was unexercised)
  ALIVE: surface always unknown     (DefinitionChange.surface asserted nowhere)
-> after 4 new tests: KILLED 18/18
```

Round 2, 28 mutants across the engine **and** the CLI (merge-base resolution, first-run detection, the stale-alert check, the mutual-exclusion guard):

```
KILLED 26/29 -> 27/28 -> KILLED 28/28
  ALIVE: drop diff-local ambiguity guard  -> guard DELETED: provably implied
                                             by the global uniqueness check
  ALIVE: no suffix trim in render         -> first assertion did not
                                             discriminate; sharpened
  ALIVE: first_run from truthiness        -> exit code passed for the wrong
                                             reason; now asserts the message
```

Each round changed the patch, not just the tests. The `_FIELD_CLASS` fallback is the one that looked like dead code and is not: it exists so a **stale lock naming a field the current format no longer emits** still diffs instead of raising `KeyError`, now `test_lock_field_retired_from_the_format_degrades_to_annotation`.

Harnesses at `/tmp/mutate_r4.py` and `/tmp/mutate_r4b.py`; both restore the sources in all paths.

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

Code review ran in a subagent and found **four ship-blockers**, all one root cause: `_last_change` was derived from the lock on disk and validated by nothing. The reviewer's summary — "the diff engine underneath is good work; the problems are concentrated in the reporting surface, which is the entire deliverable" — was correct, and the first finding is the worst kind.

- **Finding 1 (HIGH): the committed lock shipped two fabricated high-severity alerts.**
  - The first version of `_last_change` claimed `renamed execution_load -> cortex_exec_step_load` and a consumer change on `orion:verb:result`. `git show --name-only` on that commit touches **zero** registry files. It was residue of my own mutation test: I locked a deliberately-mutated registry state, reverted it, and re-locked, so the block recorded the *revert* as though it were the change.
  - **Fix:** `_last_change` is now derived from the **merge base**, not the working copy's lock, and the lock was regenerated from a genuine base.
  - **Evidence:** block now reads `"initial lock -- no prior state to diff against"` with `base: merge base f835c737c (origin/main) has no lock yet`.
  - This is the exact class of misinformation the feature exists to prevent, shipped inside the feature. It is also why finding 2 matters.

- **Finding 2 (HIGH): `_last_change` was last-write-wins, so earlier real deltas vanished while the gate stayed green.** Two `--update` calls on one branch and the first call's high-severity consumer removal was gone.
  - **Fix:** the block answers one fixed question — "what does this branch change relative to the merge base?" — which is idempotent under repeated `--update`.
  - **Evidence:** `test_repeated_updates_report_the_cumulative_branch_delta` drives two edits through two `--update` calls and asserts **both** survive; `test_update_is_idempotent` asserts byte-identical output.

- **Finding 3 (HIGH): a lock clobbered to `{}` read as "first run", so `--update` wrote "initial lock" and discarded every real delta, gate green.** `first_run = not whole` conflated *falsy content* with *no file*.
  - **Fix:** `_load_lock` returns the file's actual existence.
  - **Evidence:** `test_clobbered_lock_is_not_treated_as_a_first_run`.

- **Finding 4 (HIGH): "an agent cannot re-lock quietly" was false.** Nothing compared `_last_change` to anything, so hand-editing it to `[]` passed the gate and every test.
  - **Fix:** `--gate` now recomputes the block from the merge base and fails on disagreement. That turns the sentence from a convention into a constraint. Also added `fetch-depth: 0` to the CI checkout so `git merge-base` resolves — without it the check would have degraded silently in the one place it matters most.
  - **Evidence:** `test_gate_fails_when_the_alert_block_was_hand_edited` (rc 1); live probe hand-editing the real lock produced `committed: ['nothing to see here'] / expected: ['initial lock...']` and FAIL.
  - When the base genuinely cannot be resolved the check is skipped **with a printed note**, asserted by `test_gate_skips_block_verification_loudly_when_base_is_unresolvable`.

- **Finding 5 (MEDIUM): the most common edit rendered as `A… -> A…`.** `_short` truncated both sides independently at 60 chars, so amending the tail of a long `meaning` — 29 of 38 field-channel meanings exceed 60 chars — displayed two byte-identical strings while claiming a change.
  - **Fix:** `_render_pair` strips the shared prefix and suffix and centres the window on the differing region.
  - **Evidence:** three tests (tail, head, mid-string). The mid-string one asserts `"B" * 20 not in left` — an earlier version asserted on `…` counts and **failed to kill the no-suffix-trim mutant**, because the truncation path emits one too.

- **Finding 6 (MEDIUM): I claimed a mispaired rename was "purely cosmetic". It is not.** Severity survives, but "renamed A → B" tells the reader the metric still exists and needs no consumer migration — the inverse of the truth when A was deleted, which is the R4 motivating incident itself.
  - **Fix:** pairing now requires the definition to be unique **across the whole lock**, not just within the diff. Given the 49.7% collision rate, diff-local uniqueness was never enough.
  - **Evidence:** `test_pairing_requires_uniqueness_across_the_whole_lock` (a surviving node with the same definition blocks pairing) and `test_distinctive_definition_still_pairs_with_a_survivor_present` (real renames still pair). The false claim is retracted in-place in the docstring.

- **Finding 7 (MEDIUM): a real routing change emitted no alert at all.** `feeds_dimensions` packs `self_state_dimension` and `evidence_dimension` into one *positional* tuple, and `fingerprint()` sorted it — so swapping which dimension a channel feeds was a zero-delta edit.
  - **Fix:** `ORDERED_FIELDS`; sorting now applies only to genuine sets.
  - **Evidence:** `test_feeds_dimensions_is_positional_not_a_set` (high) plus `test_set_valued_routing_fields_are_still_order_insensitive` (the property that keeps reordered YAML quiet).

- **Finding 8 (MEDIUM): `make ... UPDATE=1 GATE=1` emitted `--update--gate`.** Fails safe, but adding a space would have been *worse* — `--update` returns before the gate block, so it would rewrite the lock and exit 0 with the gate silently skipped.
  - **Fix:** rejected in `main()` as mutually exclusive; space added so the error is legible.
  - **Evidence:** `test_update_and_gate_are_mutually_exclusive` asserts exit 2 **and** that no lock was written.

- **Findings 9–12 (MEDIUM/LOW):** `stability`/`kind` are unreachable above medium because they live inside a free-text `notes` string — documented as a known gap under `SEVERITY`, since lifting them requires a `lineage.py` change with its own blast radius. Field-channel `level` lists are joined unsorted, so my absolute "reordering produces no delta" claim was wrong — corrected in the module docstring with the exception named. Dead `HIGH_SEVERITY_KINDS` deleted (CLAUDE.md 0A). `_write_lock` now writes with an explicit encoding.

- **Not a finding, but found while fixing:** `Path.relative_to` raises for any path outside the repo, so a redirected `LOCK_PATH` produced a `ValueError` traceback instead of a report. Routed through `_rel()`.

### Re-verification after the fixes

```
pytest tests/test_metric_definition_drift.py -q          45 passed
pytest (all three metric-layer files) -q                114 passed
mutation harness, 28 mutants                       KILLED 28/28
/tmp/r4venv (pydantic, pydantic-settings, PyYAML)   gate: PASS
```

The second mutation round found three survivors, and each one changed the patch rather than just the tests: the diff-local ambiguity guard was **provably unkillable because the global guard implies it**, so it was deleted rather than tested around; the suffix-trim assertion was sharpened after the first version failed to discriminate; and a missing-lock test now asserts the *message*, since the exit code alone passes for the wrong reason.

## Restart required

```text
No restart required.
```

Read-only static gate. No service reads the lock at runtime.

## Risks / concerns

- **Severity: low.** An agent can still run `--update` and commit without reading the deltas. What it *cannot* do is make them absent from the diff, or edit them after the fact — the gate recomputes the block from the merge base. The mechanism is enforced disclosure, not consent. Stated because it is the honest ceiling on this design.
- **Severity: low.** The lock is 247KB. Git handles it and diffs only show changed entries, but it is the largest generated artifact in `config/`.
- **Severity: low.** `organ_signal` definitions are thin (`organ=X class=Y`), so 49.7% of nodes share a fingerprint — quantified above with why it does not weaken the alert. Enriching the organ registry would improve rename precision and is not in scope here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1666
