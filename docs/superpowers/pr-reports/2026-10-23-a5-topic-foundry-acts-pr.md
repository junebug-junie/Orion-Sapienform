## Summary

- Each topic-foundry training/labelling run is now one act in the act ledger: actor `topic_foundry.run`, verb `maintain` (body economy Patch A5, spec `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`, "Patch A results" producer split item 4).
- The act opens when `_run_training` starts work. It closes once: 1 on `status=complete`, 0 on failure.
  - Failures include "No documents available for training", which was 37 of 38 failures in the last 7 days.
  - Any exception closes it at 0. Anything that escapes both paths (a BaseException) also closes it at 0, through `finally`.
- **The GPU cost join is not possible yet, and this is reported rather than forced.**
  - Each LLM call keeps its own envelope id: 17,449 leases and 17,449 distinct ids over 10-16..23. orion-llm-gateway leases under that envelope id.
  - Sharing one id across a run's ~200 sequential calls was tried, and review rejected it. orion-bus-mirror would record each gateway reply → next request as a cross-organ causal hop, with think time as its latency, for up to an hour per run. That pollutes the hop baselines of `bus_synaptic_prediction_error`.
  - The act still carries a run-derived correlation id. Follow-up: the gateway forwards an explicit turn id to the pool.
- New flag `TOPIC_FOUNDRY_ACT_EMIT_ENABLED`, on by default. `orion:grammar:event` now lists `orion-topic-foundry` as a producer.

## Outcome moved

The maintenance family gets a graded act per run, with wall seconds. Once live, it also gets GPU slot-seconds.

Live rate (topic_foundry_runs, last 7 days to 2026-10-23): 89 runs, 51 complete and 38 failed. That is about **12.7 acts/day**.

## Current architecture

- Hub's scheduler, or a manual `POST`, starts a run. orion-topic-foundry runs `_run_training` as a FastAPI BackgroundTasks job in a worker thread.
- The run writes `topic_foundry_runs`, segments and artifacts, then publishes `topic.foundry.run.complete.v1`.
- The LLM calls go over the bus to the LLM gateway (`app/services/llm_client.py`). Each call is run through `_run_blocking`, sometimes in a fresh thread.
- No act was published.

## Architecture touched

- orion-topic-foundry only, plus the `orion:grammar:event` producer list.
- The event is built in `app/services/act_emit.py`. `orion.grammar.acts.act_opened_event` was not usable: its `_provenance` only accepts orion-hub and orion-thought, and the shared module was out of scope. `build_act` is reused.
- `orion/grammar/act_ledger.py` is untouched.

## Files changed

- `services/orion-topic-foundry/app/services/act_emit.py`: new. It contains:
  - the act builder;
  - `RunActTrace` (synchronous, opens once, closes once, never raises);
  - `start_run_act`, which is guarded, so a broken act build falls back to no trace;
  - a sync bus publish.
- `services/orion-topic-foundry/app/services/training.py`: opens after `status=running`; closes on complete, in `except` and in `finally`.
- `services/orion-topic-foundry/app/settings.py`, `.env_example`, `docker-compose.yml`: `TOPIC_FOUNDRY_ACT_EMIT_ENABLED=true`.
- `services/orion-topic-foundry/README.md`: section "Act ledger: each training run is one act".
- `services/orion-topic-foundry/tests/test_training_run_act.py`: new, 8 tests.
- `orion/bus/channels.yaml`: adds `orion-topic-foundry` to the producers.
- `config/metrics/metric_definitions.lock.json`: re-locked (routing_changed).

## Act contract

- **Key:** the run id, so `act_id = stable_act_id("topic_foundry.run", "maintain", run_id)`.
  - Open and close are once-guarded. The writer is also first-close-wins.
  - A restart in the middle of a run (any deploy) leaves the act open. `run_recovery` reaps the run row at startup, possibly as `complete` if it has segments. The writer expires the act at 0 at its horizon. Deploy-interrupted runs grade 0 on purpose.
- **Expect:** `artifact_produced` within 3600 s.
  - That is the p90 of complete runs from start to completion over 30 days (2335 s, n=256), rounded up to the hour. This is the same rule as the hub's reading act. The run has no deadline of its own.
  - Measured: p50 221 s, p99 3143 s, max 3640 s. The longest runs exceed the horizon and grade 0 (late), by design.
- **Outcome:** 1 when the run reaches `status=complete`, which means segments are written, topics are labelled and artifacts are recorded. Otherwise 0.
- **Cost:** `ActCostV1(seconds=wall)`.
- **correlation_id:** `uuid5(NAMESPACE_URL, "orion:topic_foundry.run:<run_id>")`. No lease carries it yet, so the GPU cost fill finds nothing. See Summary.

## Schema / bus / API changes

- Added: orion-topic-foundry publishes `grammar.event.v1` `act_opened`/`act_closed` on `orion:grammar:event`.
- No other behavior change. The LLM calls are unchanged.

## Env/config changes

- Added keys: `TOPIC_FOUNDRY_ACT_EMIT_ENABLED` (default true).
- `.env_example` updated: yes.
- Local `.env` synced: `python scripts/sync_local_env_from_example.py --all-keys orion-topic-foundry`.
  - Added `TOPIC_FOUNDRY_ACT_EMIT_ENABLED='true'`.
  - Also added the previously missing `HEARTBEAT_INTERVAL_SEC='10'`.
- Skipped keys requiring operator action: `TOPIC_FOUNDRY_DRIFT_DAEMON` is diverged (local `true`, example `false`). This divergence predates this PR and was left alone.

## Tests run

```text
topic-foundry has no CI job and its deps (hdbscan/joblib/umap) are not in the repo venv, so the suite ran inside the
service image with this worktree mounted:
  docker run --rm --network none -v <worktree>:/w:ro ... orion-topic-foundry-topic-foundry python -m pytest tests -q
  -> 154 passed, 4 skipped, 1 failed: tests/test_chat_corpus_builder_stages.py::test_stage_pipeline_outputs_expected_records
     fails identically on main (736af1b56), unrelated.
  tests/test_training_run_act.py: 8 passed (after review fixes).
CI static gates run locally (same 22 as PRs #2886/#2889) -> 22/22 PASS
```

## Evals run

```text
None. orion-topic-foundry has no evals/ for runs. The live evaluation is graded closes per family in grammar_acts.
```

## Docker/build/smoke checks

```text
Tests ran in the existing service image (above). No dependency change, no rebuild needed for the test.
```

## Review findings fixed

- **SHOULD.** A shared envelope correlation id across a run's calls would make orion-bus-mirror record fake causal hops for up to an hour per run, which feeds `bus_synaptic_prediction_error`.
  - Fix: reverted. Every call keeps its own id, and `llm_client.py` is unchanged from main. The cost join is reported as not possible until the gateway forwards a separate turn id.
- **SHOULD.** Building the act after `status=running` was written, and outside the try, could strand a run in `running`.
  - Fix: `start_run_act` is guarded and falls back to a no-op trace.
  - Evidence: `test_a_broken_act_build_never_strands_the_run`.
- **NIT.** A deploy mid-run grades the act 0 while `run_recovery` may mark the run `complete`.
  - Fix: documented as intended in `act_emit.py` and the README. Closing from the reaper could create close-only rows for runs that never opened an act (queued runs, enrichment-only passes).
- **NIT.** The horizon sits below the observed max.
  - Fix: the basis text now says so.
- **NIT.** The publish-failure test did not prove the close was attempted.
  - Fix: it now records both attempts.
- **NIT.** The fresh-thread path was untested.
  - Fix: moot. The shared-correlation code was removed.
- **NIT.** The lock file may conflict with other PRs.
  - Fix: re-lock after rebasing (noted under Risks).

## Restart required

```bash
scripts/safe_docker_build.sh orion-topic-foundry up -d --build
```
Run this from the clean primary `main` checkout after the merge.

## Risks / concerns

- **Severity: medium.** This family has no GPU cost until orion-llm-gateway forwards an explicit turn id to the pool, separate from the envelope id. **Mitigation:** follow-up PR in orion-llm-gateway (proposal: `options.turn_correlation_id`, falling back to the envelope id). Then pass `run_correlation_id(run_id)` there.
- **Severity: low.** The other A5 PRs touch the same `channels.yaml` list and lock file, which may conflict on merge. **Mitigation:** the producer entry sits at a different list position. Re-run `check_definition_drift.py --update` after rebasing.
- **Live check:**
  ```sql
  SELECT act_id, outcome_value, cost_json
  FROM grammar_acts
  WHERE actor = 'topic_foundry.run'
  ORDER BY opened_at DESC
  LIMIT 5;
  ```

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
