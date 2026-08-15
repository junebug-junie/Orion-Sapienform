# PR report — the lineage layer counted producers as consumers and missed generic reads

Branch: `fix/metric-consumer-blindspots`

## Summary

- Three defects in the metric semantic layer's **blast-radius** path, all found by acting on its output and being wrong.
- **Writes counted as reads.** `visit_Subscript` ignored AST context, so `vec["m"] = x` scored as a consumer. A channel's own producer appeared inside its blast radius.
- **Generic consumers were invisible.** A function that takes a whole channel vector and aggregates it reads every channel while naming none — undetectable by a string-literal scan.
- **There was no producer concept at all.** The layer resolved *declared* producers and *discovered* consumers, and never a write site.
- New `--generic-consumers` and `--unwritten` reports, plus `make` targets for both.

## Outcome moved

The card for `field_coherence_warning` used to say five call sites and imply nothing read it. It now separates producer from consumers and refuses to let a retirement decision rest on the blast radius alone:

```
  WRITTEN BY (discovered, non-test): 2
      services/orion-field-digester/app/worker.py:272  [subscript_write]
      services/orion-field-digester/app/worker.py:282  [subscript_write]
  BLAST RADIUS (discovered, non-test, high-confidence): 3
      orion/field/pressure.py:49  [collection_member]
      ...
  PLUS 17 generic whole-vector consumer(s) (13 confirmed) that read this
  channel without naming it.  Blast radius above is a FLOOR, not a total --
  do not retire this channel on it alone.
```

The consumer that made the original mistake possible — `_current_pressure_proxy(vector: dict[str, float])` in `orion/attention/field_attention/selectors.py`, a `max()` over every channel that feeds attention target selection — is now discoverable. Acting on the old output under CLAUDE.md 0A's "kill means kill" would have deleted a live channel's producer.

## Scope correction against my own earlier claim

I said some share of the 194 orphans were false. **They are not.** All 194 are `bus_channel`/`inner_state`/`organ_signal`, none of which live in node vectors, so no field channel is in that list at all. Verified independently by running the pre- and post-patch scanners side by side: 20 tokens lost high-confidence hits, **0 dropped to zero**, and the gate output is unchanged at `{bus_channel: 17, inner_state: 13, organ_signal: 164}`, `PASS`.

The damage was confined to blast radius — which is precisely the tool a retirement decision uses.

## Files changed

- `orion/metrics/consumers.py`: Load/Store split; `KIND_CHANNEL_KWARG`, `KIND_FIELD_KWARG`, `WRITE_KINDS`; `ConsumerHit.callee`; `producers_for()`.
- `orion/metrics/generic_consumers.py` (new): two-tier whole-vector consumer detection.
- `scripts/check_metric_lineage.py`: producer section on the card, the FLOOR warning, `--generic-consumers`, `--unwritten`.
- `Makefile`: `check-metric-generic-consumers`, `check-metric-unwritten`.
- `tests/test_metric_generic_consumers.py` (new): 39 tests.

## Honest labels, checked rather than assumed

Every reliability label here was measured, and two of them contradicted what I first wrote:

- **`field_channel` is `MIXED`, not `STRONG`.** `cpu_pressure` has a literal `Perturbation(channel="cpu_pressure")` at `state_deltas.py:129`; `cortex_exec_step_load` has **no literal write anywhere in Python** despite being declared, decayed, merged and read. An entry on the unwritten list means "go find the writer", not "there is none".
- **`bus_channel` is `NOT ASSESSABLE`**, held out of the headline count. It came back 260 of 260 — channels publish positionally or from config, never as a named write, so the detector was measuring itself.

## Schema / bus / API changes

None. Read-only static analysis over existing registries.

## Env/config changes

None.

## Tests run

```
pytest tests/test_metric_generic_consumers.py -q          39 passed
python scripts/check_metric_lineage.py --gate             PASS  (17/13/164)
make check-metric-generic-consumers                       13 confirmed, 4 likely
make check-metric-unwritten                               96 assessable
```

## Evals run

Mutation testing, twice. The first round is the reason the review below found three MUSTs — half the mutants survived:

```
round 1 (27 tests):
  SURVIVED  module_touches_vectors always True
  SURVIVED  drop the FieldStateV1 half of the heuristic
  SURVIVED  VECTOR_ATTRS loses capability_vectors
  SURVIVED  AGGREGATORS emptied
  CAUGHT    revert Load/Store split, CHANNEL_KWARG_NAMES emptied, field_kwarg removed
```

Each survivor was a real hole, not a harness artifact: `module_touches_vectors` was the one line the module docstring called "doing real work" and the only one with no test behind it; and `test_max_over_an_annotated_vector` matched on `.items()` before ever reaching `AGGREGATORS`, so all 11 aggregator builtins were untested despite a test named for them.

A harness note worth recording: the first background run was killed by a timeout **mid-mutation** and left `_callee_name` returning `None` on disk. The tests caught it immediately, but the harness now restores under `atexit` + a `SIGTERM` handler rather than only at end-of-loop.

## Review findings fixed

Code review ran in a subagent and found three MUSTs and two SHOULDs. It independently verified the four factual claims in the commit message and confirmed no regression from the Load/Store split.

- **MUST-1 — a leaked loop variable suppressed the FLOOR warning on exactly the tokens most likely to be misread.** `if node.surface in VECTOR_SURFACES` used whatever `node` the per-node print loop left bound, i.e. the *last* node for the token. For `confidence`, `memory_pressure` and `repair_pressure` — the three multi-surface tokens named in `gate.py`'s own docstring — the last node is not the field channel, so the warning silently never printed.
  - **Fix:** `any(n.surface in VECTOR_SURFACES for n in nodes)`.
  - **Evidence:** all three now print the warning; a mutant reverting it is killed.
  - This reintroduced the exact incident the patch exists to prevent, through the back door, and no test covered `cmd_metric`.

- **MUST-2 — `KIND_FIELD_KWARG` was 87% of all write evidence and mostly noise.** Any kwarg *named* after a metric counted as writing it. `confidence=` appears 425 non-test times across 142 callees and **not one targets a schema that declares `confidence`** — producing a 434-row "WRITTEN BY" list with zero real writers, and holding 19 metrics off `--unwritten` on passthrough like `max_gap_sec=args.max_gap_sec`.
  - **Fix:** `ConsumerHit` now carries the callee, and `field_kwarg` counts only when the callee is a schema declaring the metric. No `schema_ids` supplied means no `field_kwarg` evidence — the conservative direction, which can leave a metric *on* the unwritten list but never silently take one off.
  - **Evidence:** `confidence` writers 434 → 9; `--unwritten` 85 → 96 as the falsely-suppressed metrics reappear.

- **MUST-3 — the mutation gap above.** Fixed with tests for `module_touches_vectors` itself, the aggregator path, and `capability_vectors` (which contributes 5 of the 13 confirmed live sites and previously broke no test when removed).
  - Re-running the harness after the fixes gave **16/17, and the survivor was MUST-1's own fix** — reverting the leaked-variable correction failed no test, because nothing covered `cmd_metric` at all. So the fix for the review's top finding was itself unverified until the harness said so. `test_floor_warning_prints_for_a_multi_surface_token` closes it, and asserts its own precondition (that the last node is NOT the field channel) so it cannot pass vacuously. Final: **17/17.**

- **SHOULD-4 — 2 of 6 `likely` results were false positives of exactly the class the docstring claimed the filter removed.** `module_touches_vectors` was a substring search, so 20 of 58 matching modules matched on comment/docstring prose with zero AST reference — this detector's own module among them.
  - **Fix:** real AST reference check.
  - **Evidence:** `likely` 6 → 4, and the survivors are exactly the four the reviewer independently judged genuine. The two removed were a rank-score dict and a prediction-error-domain dict.

- **SHOULD-5 — the "KNOWN MISSES" list understated the gap.** `helper(vector)` and `return vector` — two of the commonest ways a vector escapes — were not listed. Now listed, along with `{**vector}`, `vector | other`, and `vector.get(k)` in a loop, each probed rather than assumed.

- **Nits fixed:** line-cite `state_deltas.py:87` → `:94`; "first generic use" now says `ast.walk` (breadth-first) order, not source order; `--unwritten` counts *scan tokens*, not URNs (595 URNs collapse to 397 tokens); the "every 2-second tick" claim is now marked as config truth (`ATTENTION_POLL_INTERVAL_SEC=2.0`) rather than observed cadence; the file is parsed once instead of read twice; `make` targets added for both new reports.

- **Accepted, not fixed:** `vec["m"] += 1` is genuinely both a read and a write and is now classified write-only. Repo-wide there are **0** augmented-assign and **0** `del` sites across all 397 tokens, so nothing is lost today; a future `vec["cpu_pressure"] += delta` would become invisible blast radius. Noted in the test docstring rather than solved.

## Restart required

```text
No restart required.
```

Read-only static analysis. No service reads any of this.

## Risks / concerns

- **Severity: low.** The `likely` tier is a shape plus a neighbourhood, not a dataflow proof. It is reported separately and never gates an orphan verdict — `surfaces_at_risk()` requires a `confirmed` site, so a heuristic cannot silently erase an orphan.
- **Severity: low.** An unannotated vector parameter is still invisible. A clean result means "none found", never "none exist", and the report says so in those words.

## PR link

<pending>
