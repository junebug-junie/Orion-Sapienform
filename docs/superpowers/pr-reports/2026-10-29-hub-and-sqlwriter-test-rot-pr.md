# Fix hub and sql-writer test rot, and run both folders in CI

## Summary

- The full Hub test folder had 33 failing tests on main. The full sql-writer test folder had 14. CI did not see them, because CI ran only hand-picked files from each folder.
- Two failures were real product bugs. Both are fixed in product code:
  - sql-writer dropped two biometrics fields (`peak_pressure`, `peak_pressure_channel`) from every per-node row since 2026-08-15. The table had no columns for them.
  - The Hub memory-graph "validate" preview returned 503 for every draft when the memory database pool was missing. Only consolidation drafts need that pool.
- One failure was a false bus contract: `channels.yaml` named sql-writer as a reader of curiosity help requests, but sql-writer never reads them.
- The other failures were old test expectations, tests that read the operator's real files, and tests that depended on run order. Each is fixed in the test. No test was deleted.
- One security question is open. Three recall-canary routes have answered without an operator token since 2026-05-01. A new test proves this and is marked "expected to fail" until Juniper decides (see Risks).
- CI now runs the whole `services/orion-sql-writer/tests` folder and the whole `services/orion-hub/tests` folder. Both jobs trigger on any change under `orion/**` or `config/**` too.

## Outcome moved

- `services/orion-sql-writer/tests`: 14 failed -> 0 failed (1132 passed, 175 skipped).
- `services/orion-hub/tests`: 33 failed -> 0 failed (4408 passed, 143 skipped, 1 expected failure).
- Both folders are green in a fresh virtual environment with an empty home folder, the same as a CI runner sees.
- A test that breaks anywhere in either folder now fails a pull request.

## Current architecture

- sql-writer writes each bus payload into a table. Its row writer silently drops any payload field that has no table column.
- Hub tests share one `conftest.py`. Before every test, it removes the Hub's `scripts.*` and `app.*` modules from Python's module cache. A test that imported a route at the top of its file therefore holds an old copy, and a patch made by name lands on a new copy.
- CI: `orion-sql-writer-tests.yml` listed 33 sql-writer test files by name. About 20 Hub workflows each listed a few Hub test files.

## Architecture touched

- sql-writer: biometrics summary table model, boot-time column adds.
- Hub: memory-graph validate route.
- Bus contract: `orion/bus/channels.yaml` (one consumer removed).
- CI: one workflow changed, one workflow added.

## Every failure, its class, and the fix

Classes: (a) real product bug, (b) stale test, (c) host or environment leak, (d) run-order dependence.

### sql-writer (14)

| Test | Class | Fix |
|---|---|---|
| `test_biometrics_summary_sql_shape::test_every_payload_field_has_a_column` | a | Add `peak_pressure` / `peak_pressure_channel` columns, boot-time `ALTER ... ADD COLUMN IF NOT EXISTS`, and `scripts/sql/2026-10-29_biometrics_summary_peak_pressure.sql`. New tests guard the boot ALTER, nullability, and the real `_write_row` keeping both fields (fails if the columns are removed). |
| `test_subscription_matches_declared_consumers::test_no_new_declared_consumer_channel_goes_unsubscribed` | a (contract) | sql-writer has no route, model, or table for `HelpRequestV1`. Removed it from the channel's `consumer_services`. |
| `test_biometrics_induction_timestamp_format::test_writer_does_not_serialise_the_timestamp_as_iso_json` | b | The row-building line became a chain (0a59d3712). The test now finds the generic arm of the chain and still checks it has no `mode="json"`. |
| `test_chat_history_response_identity_merge` x4 | b | The fake database session now accepts the row-lock call `execute(stmt, params)` and the table lookup `query()`. One test also checks that the lock is taken. |
| `test_journal_entry_payload_boundary::test_journal_entry_write_uses_nested_payload...` | b | The writer now also writes `evidence_units` rows after the journal row. The test records every write and checks the first one. |
| `test_gpu_pool_state_history::test_subscription_route_retention_and_catalog_agree` | d | `test_vision_persistence_lane.py` removed all `app.*` modules at import time. Removed that. |
| `test_notify_attention_ack` x3, `test_notify_attention_escalate` x2 | d (showed as c) | Same cause. The stale module made these tests connect to the real database host. |

### Hub (33)

| Test | Class | Fix |
|---|---|---|
| `test_memory_graph_routes::test_memory_graph_validate_fixture_roundtrip` | a | Validate needs the pool only when `consolidation_draft_id` is set. Two new tests: a plain draft validates with no pool; a consolidation draft still returns 503 with no pool. |
| `test_agent_trace_debug_panel::...scroll_lock...` | b | `cockpitHudRoot` now follows `agentTraceModal`, so no trailing `;`. |
| `test_biometrics_view_ui::...deactivates_biometrics_view...` | b | The check window now ends at the next panel toggle, not at 700 characters. |
| `test_chat_history_no_raw_publish::test_ws_turn_still_persists_the_turn` | b | The turn now also opens a `say` act on the grammar channel (act ledger). Checked apart from the three chat-history publishes. |
| `test_field_channel_glossary_routes::...38_raw_channels...` | b | Perception "rest" text was reworded; the caveat is now "is NOT a reading". |
| `test_handle_chat_request_turn_effect::...retries_memory_graph_suggest_on_timeout` | b | Every Hub request now defaults to the quick route, so the timeout retry to quick does not fire. The test now asserts one call on quick and no retry. A new direct test covers the retry branch itself. |
| `test_hub_ui_polish::test_hub_main_layout_is_fifty_fifty...` | b | The 50/50 split moved from a `md:w-1/2` class to CSS (`.hub-ekg-drawer { width: 50%; }`). |
| `test_memory_graph_bridge_ui` x2 | b | The fixed timeout constant became a resolver function. |
| `test_memory_graph_suggest_coalesce_ui::test_coalesce_prefers_appendix_c_json` | b | The coalescer always adds an `utterance_text_by_id` map. |
| `test_mind_hub_tab::test_mind_runs_modal_is_sibling...` | b | A GPU-leases modal now sits between the two. The check now parses the HTML and compares parents, instead of matching exact closing tags. |
| `test_recall_strategy_profiles_runtime` x4 | b | The recall-canary routes no longer take an operator token (7495edab7). Calls updated. |
| `test_recall_strategy_profiles_runtime::...accepts_http_only_cookie_operator_token` | b, open | It could not fail while no guard exists. Replaced by `test_recall_canary_routes_refuse_a_request_without_operator_token`: an HTTP test on all three routes that expects 403. Marked `xfail(strict=True, raises=AssertionError)`. Today it gets 400/404, not 403. See Risks. |
| `test_substrate_review_runtime_hub_debug` x2 | b | Substrate tab button falls back to the legacy link; tab key is `effectiveTab`. `window.OrionHub` check uses a word boundary (`window.OrionHubOpenOrganSignals` is unrelated). |
| `test_turn_orchestrator_utterance_origin::...mind_appraisal_text...` | b | A help-request kickoff can carry the role-teach advisory in front. The test checks the kickoff is intact at the end. |
| `test_memory_graph_consolidation_draft_approve` x3 | b + d | Approve now announces each card to the crystallizer; stubbed and checked. Router imported per test. |
| `test_fcc_model_labels_api::...reads_fixture_env` | c + d | The route read the operator's real `~/.fcc/.env` and the live FCC server, because the test patched a different settings copy. App imported per test. |
| `test_memory_consolidation_draft_routes` x6 | d | Router imported per test so the name-based patches reach it. |
| `test_substrate_effect_endpoint` x3 | b + d | Pre-turn appraisal is on by default since 2026-10-10, so the legacy pipeline these tests drive did nothing. Pinned off with the existing `pre_turn_appraisal_off` fixture; modules resolved per test. |
| `test_substrate_effect_pipeline::...internal_failure_without_raising` | d | Calls the pipeline through the patched module, so the failure path is exercised. |

### Found while widening CI (passed on this host, failed on a clean runner)

| Test or file | Class | Fix |
|---|---|---|
| `test_fcc_claude_bridge_run` x5 | c | They rendered a real MCP config from the operator's `~/.fcc/.env` (with `GITHUB_PAT`). They failed with no such file. MCP rendering is now stubbed in this file. |
| `test_chat_route_tagging::test_classic_planrunner_path_tags_chat_route` | d | Pins `scripts.main.bus` to `None`, which the test assumed. |
| `evals/zelda_experience_replay.py` (via `retro/test_experience.py`) | d | Sets the Hub's `app.*` modules aside while it imports orion-prediction's `app` package, then puts them back. |
| `test_hub_ui_polish._render_hub_index`, `test_memory_graph_routes` | d | Module removals now restore or only target a foreign `scripts` package. |
| `test_observability_api::test_api_signals_trace_invalid_id_returns_400` (first CI run) | c | `with TestClient(app)` ran startup, which sets the patched cache to `None` when the bus is unreachable. Passed only on a host with a live bus. Now no startup. |
| `test_turn_timer_js`, `test_draft_first_display` node cases (first CI run) | c | Newer Node prints the "spec" report by default, so the `# pass N` lines were missing. Now `--test-reporter=tap`. |

## Files changed

- `services/orion-sql-writer/app/models/biometrics_summary.py`, `app/main.py`, `scripts/sql/2026-10-29_biometrics_summary_peak_pressure.sql`: new columns.
- `services/orion-hub/scripts/memory_graph_routes.py`: validate pool only for consolidation drafts.
- `orion/bus/channels.yaml`: removed a false consumer.
- `services/orion-hub/evals/zelda_experience_replay.py`: module isolation.
- Test files listed in the tables above.
- `.github/workflows/orion-sql-writer-tests.yml`, `.github/workflows/orion-hub-full-suite.yml`: run whole folders.

## Schema / bus / API changes

- Added: `orion_biometrics_summary.peak_pressure DOUBLE PRECISION`, `orion_biometrics_summary.peak_pressure_channel TEXT` (both nullable). The writer adds them at boot.
- Removed: `orion-sql-writer` from `consumer_services` of `orion:curiosity:help:request`. No subscription existed, so nothing at runtime changes.
- Behavior changed: `POST /api/memory/graph/validate` no longer returns 503 for a plain draft when the memory pool is missing. With `consolidation_draft_id` it still does.
- Compatibility notes: the columns are additive and nullable. Existing rows keep NULL.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-sql-writer pytest -q services/orion-sql-writer/tests
  -> 1132 passed, 175 skipped                          (host venv)
  -> 1132 passed, 175 skipped                          (fresh venv, CI deps)
exact CI step "Run sql-writer unit tests" (fresh venv)
  -> 1240 passed, 175 skipped
PYTHONPATH=.:services/orion-hub pytest -q services/orion-hub/tests
  -> 4408 passed, 143 skipped, 1 xfailed               (host venv)
  -> 4411 passed, 140 skipped, 1 xfailed               (fresh venv, empty HOME)
after review fixes, touched files:
  test_recall_strategy_profiles_runtime.py -> 14 passed, 3 xfailed
    (--runxfail: each fails on the 403 assertion, got 400/404)
  test_handle_chat_request_turn_effect.py  -> 14 passed
  test_biometrics_summary_sql_shape.py     -> 9 passed (new test fails with columns removed)
python scripts/check_env_template_parity.py            -> PASS
python scripts/check_bus_reply_channels.py             -> 0 uncovered
pytest tests/test_curiosity_peer_registry.py tests/test_curiosity_peer_schema.py -> pass
```

## Evals run

```text
PYTHONPATH=. python services/orion-hub/evals/zelda_experience_replay.py -> PASS 5/5
sql-writer evals in the CI step (handoff replay, walkway reducers, count estimator,
storage write replay, act ledger calibration) -> pass in the run above
```

No new eval: this patch fixes tests and two small product bugs. It adds no new behavior to grade.

## Docker/build/smoke checks

```text
Not run. No Dockerfile, compose, or dependency change.
The sql-writer column add is verified by test only. Live state before deploy (2026-10-29):
orion_biometrics_summary has no peak_pressure columns (information_schema query).
After the sql-writer restart, the boot ALTER adds them: UNVERIFIED until deploy.
```

## Review findings fixed

Review ran in a subagent against `origin/main...HEAD`. No blockers.

- Finding (should): the "expected to fail" cookie-token test failed on a Python call signature, not on auth, and a bare xfail accepts any error.
  - Fix: replaced with an HTTP test on all three canary routes expecting 403, `raises=AssertionError`. The unguarded routes are now listed under Risks for Juniper.
  - Evidence: with `--runxfail` each case fails on `assert resp.status_code == 403` (400, 404, 404).
- Finding (should): the sql-writer job runs the whole folder, but its triggers were still the old narrow path list.
  - Fix: added `orion/**` and `config/**` to both trigger lists.
  - Evidence: `.github/workflows/orion-sql-writer-tests.yml`.
- Finding (nit): the quick-retry branch lost its only test.
  - Fix: direct test of `_should_retry_memory_graph_quick`, both the retry and no-retry case.
  - Evidence: `test_memory_graph_quick_retry_still_fires_when_no_route_is_set` passes.
- Finding (nit): no end-to-end test of the peak-pressure fix.
  - Fix: `test_write_row_keeps_peak_pressure_fields_end_to_end` drives the real `_write_row`.
  - Evidence: it fails when the two columns are removed from the model.
- Finding (nit): the say-act check accepts zero acts. Not changed. The act is gated by `HUB_ACT_LEDGER_ENABLED` and is not what this test is about.
- Finding (nit): the boot-time ALTERs have no lock timeout. Not changed. They follow the existing `measurements` convention and only change the catalog.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary `main` checkout after merge. Then check the columns exist:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select column_name from information_schema.columns where table_name='orion_biometrics_summary' and column_name like 'peak_pressure%'"
```

## Risks / concerns

- Severity: medium (security). Concern: commit 7495edab7 (2026-05-01, message "various") removed the operator-token check from three Hub routes: `POST /api/substrate/recall-canary/query`, `.../runs/{id}/judgment`, and `.../runs/{id}/create-review-artifact`. Every sibling mutation route still checks the token. The judgment route can emit pressure. Anyone who can reach the Hub can call these three today. I could not confirm this was intended, so I did not put the check back. Mitigation: `test_recall_canary_routes_refuse_a_request_without_operator_token` is `xfail(strict=True)`. If the check comes back, it passes, strict mode fails CI, and someone must remove the mark. Juniper: restore the guard, or confirm it is intended and delete that test.
- Severity: low. Concern: the memory-graph suggest "retry on quick after a timeout" path in `api_routes._should_retry_memory_graph_quick` can run through a real request only for an unknown route name, because every request already defaults to quick. It is close to dead code. Mitigation: a direct test keeps it honest; follow-up to remove it or give it a real target.
- Severity: low. Concern: the new Hub CI job runs on every PR that touches `orion/**` and takes about 13-14 minutes on a GitHub runner (6-7 locally). Mitigation: that cost is the price of not letting the folder rot again.
- Severity: low. Concern: `tests/test_single_consumer_channels_gate.py` (repo root) fails on main for 8 request channels. It is outside this task and outside CI. Not changed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2967

🤖 Generated with [Claude Code](https://claude.com/claude-code)
