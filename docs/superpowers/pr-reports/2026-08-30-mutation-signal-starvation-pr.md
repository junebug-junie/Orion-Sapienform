## Summary

- Answers the blocking question from the self-calibration roadmap (`docs/superpowers/specs/2026-08-30-self-calibration-roadmap-and-session-handoff.md`, Part 5): **why did `substrate_mutation_*` never fire?** The answer is neither of the two options that document offered.
- The scheduler is **live**, Postgres-backed, and completes a cycle every 30s. It requires telemetry matching `invocation_surface="operator_review"` **and** `target_zone ∈ {autonomy_graph, self_relationship_graph}`. Every one of the 1,358 rows in `substrate_review_telemetry` satisfies exactly one of those conditions and never both. The intersection has been empty since 2026-07-24.
- Five weeks of logs did not show it because `signals_processed: 0` had three different causes — empty store, wrong surface, wrong zone — and all three printed the same unremarkable zero.
- Ships the **diagnosis, not the cure**. Unstarving the pipeline would make a dormant self-modification loop start firing: an invasive cognition change under AGENTS.md 0A. Three options are written up; none applied. Nothing here changes what Orion decides.
- Fixes a latent crash found while instrumenting: disabling routing proposals would have raised `ValidationError` inside the cycle lock.

## Outcome moved

A permanently-zero field that looked like a healthy idle cycle now names its own cause, live, on an endpoint. The highest-value unanswered question in the repo is answered with runtime evidence.

## Current architecture

`execute_substrate_mutation_scheduled_cycle()` (orion-hub) ticks every 30s, queries `GraphReviewTelemetryRecorder` for `operator_review` rows, filters to two zones, and hands the result to `SubstrateAdaptationWorker`. Its only other signal source, `_self_revision_signals_from_latest_self_state()`, is hardcoded to `[]` (SelfStateV1 burn, 2026-07-22).

Only two writers exist for that store repo-wide:

| site | surface | zone | live rows |
|---|---|---|---|
| `review_runtime.py:248` | from the request | from the queue item | 1,356 (`operator_review`/`concept_graph`) |
| `api_routes.py:2745` | hardcoded `chat_reflective_lane` | hardcoded `autonomy_graph` | 2 |

The satisfying combination is constructed by exactly one file in the repo — `smoke_mutation_v21.py` — which is why the smoke is green while production is empty.

## Architecture touched

Observability only. No filter widened, no producer rewired, no proposal created that would not have been created before.

## Files changed

- `orion/substrate/review_telemetry.py`: adds `query_with_attrition()`; `query()` delegates to it so there is one implementation and no drift.
- `services/orion-hub/scripts/api_routes.py`: `signal_intake` block in the cycle summary, `_publish_non_cycle_signal_intake()` for early-return paths, new read-only endpoint, `max_signals <= 0` guard.
- `orion/substrate/tests/test_review_telemetry_attrition.py`: 7 tests (new).
- `services/orion-hub/tests/test_substrate_mutation_signal_intake.py`: 12 tests (new).
- `docs/superpowers/specs/2026-08-30-mutation-signal-starvation-finding.md`: the finding (new).

## Schema / bus / API changes

- Added: `GET /api/substrate/mutation-runtime/signal-intake` (read-only, no operator token — exposes only counts and histograms already in the scheduler log).
- Added: `query_with_attrition()` on `GraphReviewTelemetryRecorder`.
- Behavior changed: `max_signals <= 0` short-circuits instead of raising.
- Compatibility: `query()` returns byte-identical results; no schema, bus channel, or payload contract touched.

## Env/config changes

None. No keys added, removed, or renamed; `.env_example` untouched, so no sync was required.

## Tests run

```text
orion/substrate/tests/                                   632 passed
services/orion-hub/tests/test_substrate_mutation_signal_intake.py   12 passed
services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py 19 passed
```

Mutation testing: **12 attempted, 12 caught**, including all three the reviewer found uncaught (slice-before-sort, ascending sort, swapped filter order).

Pre-existing failures: the hub `-k "mutation or telemetry or substrate"` selection has 12 failures on `main` at `f4a1be749`, unrelated to this patch (`test_substrate_effect_*`, `test_substrate_review_runtime_hub_debug`, `test_handle_chat_request_substrate_effect`, `test_recall_strategy_profiles_runtime`). Verified by running patched and reverted trees back-to-back with an identical collection set: **byte-identical failure lists**. That suite has genuine cross-test pollution — the 12th failure's victim floats within `test_substrate_mutation_manual_route_routing.py` and the file passes in isolation. Not introduced here; not fixed here.

## Evals run

No eval harness exists for this seam. The live endpoint below is the behavioural check; the 9 CI static gates all pass.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build   -> Image orion-hub-hub-app Built
scripts/safe_docker_build.sh orion-hub up -d   -> Container orion-athena-hub Started
GET /api/substrate/mutation-runtime/signal-intake -> 200
```

Live, on the deployed hub:

```json
{"reason": "zone_filter_rejected_all", "starved": true, "consecutive_starved_cycles": 2,
 "store_total_records": 1562, "store_matched_surface": 1560,
 "before_zone_filter": 32, "after_zone_filter": 0, "usable_zone_rows_before_limit": 0,
 "surface_histogram": {"operator_review": 1560, "chat_reflective_lane": 2},
 "zone_histogram": {"concept_graph": 1560, "autonomy_graph": 2},
 "matched_zone_histogram": {"concept_graph": 1560}}
```

The two histograms disagree on purpose: the whole-store one shows `autonomy_graph: 2`, which looks like usable signal; the pre-slice one shows none, because both rows failed the surface filter. That gap is what separates a real zone mismatch from limit truncation.

The store held 1,358 rows when the investigation began and 1,562 two hours later. The starvation is ongoing, not historical.

## Review findings fixed

- **Finding:** limit truncation was misreported as a zone mismatch — 40 usable rows behind 40 newer ones would send an operator to widen `allowed_zones`, which changes nothing.
  - **Fix:** pre-slice `matched_zone_histogram` + distinct `limit_truncated_usable_signals` reason.
  - **Evidence:** `test_limit_truncation_is_not_blamed_on_the_zone_filter`; live payload shows the two histograms correctly disagreeing.
- **Finding:** the behaviour-preservation test was a tautology (`query()` delegates to the function under test); slice-before-sort and ascending sort passed unnoticed.
  - **Fix:** assert record identity and order; make the filter-order fixture use rows failing both filters.
  - **Evidence:** those exact mutations now go red.
- **Finding:** a frozen report was served stamped with the current time; three early returns never build one.
  - **Fix:** early-return paths publish; every report carries `tick_id` and `observed_at`.
  - **Evidence:** `test_a_bailing_cycle_does_not_leave_a_stale_healthy_report`.
- **Finding:** an override cycle reset the starvation counter. **Fix:** overrides no longer publish.
- **Finding:** the histogram cap could never bind and its test was vacuous. **Fix:** cap deleted; keys come from closed Literals.

## Restart required

Already deployed and verified. To redeploy from a worktree:

```bash
bash scripts/safe_docker_build.sh orion-hub build
bash scripts/safe_docker_build.sh orion-hub up -d
```

## Risks / concerns

- **Severity: low.** Two `Counter` passes per cycle over at most `max_signals` records already in memory; no extra database round trip. Read endpoint is read-only.
- **Severity: informational.** The three ways to actually unstarve the pipeline are in the finding doc and **need Juniper's decision**. Option 2 (enqueue `autonomy_graph` review queue items) matches the original design intent; Option 1 (widen the surface filter) is recommended against for the reason `api_routes.py:2665-2675` already argues; Option 3 is retiring the pipeline outright.
- **Severity: informational.** The hub test suite has pre-existing cross-test pollution. Untouched here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1999
