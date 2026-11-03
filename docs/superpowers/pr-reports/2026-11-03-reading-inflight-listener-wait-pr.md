## Summary

- A reading run can lose its Hub while the Hub is already working on the turn. The bus reports this as "no listener ... while awaiting its reply". The run now waits out the redeploy and sends the turn again, instead of failing.
- Readings get at most 2 of these re-sends per run. After 2, the run fails exactly as it did before.
- Only `reading.turn` opts in. Every other workflow still spends an attempt on this case, because its worker may already have done real work.
- The opt-in lives in the reading workflow itself (`app/reading_graph.py:IN_FLIGHT_RESENDS = 2`), not in the shared wait code.
- Each wait is still a `run.listener_wait` event. A new `case` field (`published_to_nobody` or `lost_in_flight`) says which kind of wait it was.

## Outcome moved

Live 2026-11-03: Hub was redeployed 4 times between 18:05 and 18:50 UTC. Three readings failed terminal with `turn_exception:RpcNoListener: no listener on orion:reading:turn:request while awaiting its reply`:

- `reading-8dc3c8fd...` at 18:12:52, matching the 18:12 deploy.
- `reading-8a50afd9...` and `reading-0700dac1...` at 18:29:20, matching the 18:26 deploy.

These were confirmed in `durable_resource_events`. With this change, readings like these wait and complete. The eval replays that window: with the opt-in at 0, all 16 runs fail with this same error. With the opt-in on, all 16 complete and no attempt is spent.

## Side-effect verdict for reading turns (the safety claim)

Verified in code. A duplicate reading turn costs compute and adds trace rows. Nothing lands in memory, chat history or the graph.

- The Hub listener (`services/orion-hub/scripts/reading_turn_listener.py`) runs `execute_unified_turn(reading_only=True)` with payload `no_write=True` (`world_pulse_read_pipeline._turn_payload`). That flag skips `_publish_unified_turn_chat_history`, so no `chat_history_log` row and no chat-turn envelope are written.
- The motor is reading-only. It gets WebFetch/WebSearch with an empty MCP config (`orion/fcc/mcp_config.py`) and no Bash.
- Results land only later, when Hub consumes `reading_durable_turn`.
- What a duplicate does write: a stance `thought_decision` row (with its self-prediction bet), hub.chat grammar events, cockpit frames and harness step/lifecycle grammar. Each of these is keyed by the turn's own correlation id.
- The re-send runs under a new hold, so `turn_correlation_id` differs. A stray reply from the old turn cannot be accepted, and nothing is double-counted under one key.
- Known cost: a live Hub that is only resubscribing can look gone. The bus treats two zero counts about 5 s apart as "gone", while the Hub resubscribe backoff can last up to 60 s. In that case the first turn keeps running. The re-send, which comes at least 60 s later (10 s for urgent runs), can then run beside it, and once the hold is released that first turn runs outside GPU admission. The failure path before this change did the same thing later, through the seed queue's retry. The cap of 2 bounds it.

## Current architecture

PR #3053 made `RpcPublishedToNobody` wait without spending an attempt: backoff 60/120/240/300 s, at most 6 waits, urgent runs 2 short waits, deadline-aware, counted in `listener_waits`. A plain `RpcNoListener` (the in-flight case) took the normal failure path everywhere. For reading, that path is a terminal `failed`.

## Architecture touched

- `services/orion-durable-runs` only. There are no bus, schema or env changes.

## Files changed

- `services/orion-durable-runs/app/admitted_graph.py`:
  - `listener_absence()` classifies both cases.
  - `listener_wait(..., in_flight_resends=0)` gives the in-flight case its own counter (`listener_in_flight_resends`) and cap: `min(DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES, in_flight_resends)`.
  - Backoff, urgent and deadline rules are the same as the published-to-nobody wait.
  - The event and the hold carry `case`.
  - `published_to_nobody()` keeps its contract.
- `services/orion-durable-runs/app/reading_graph.py`:
  - Adds `IN_FLIGHT_RESENDS = 2` with the safety rationale.
  - Declares the state key.
  - Passes the opt-in.
- `services/orion-durable-runs/tests/test_listener_wait.py`:
  - Covers classification, the in-flight wait and completion, the cap of 2 then failure, separate counters, backoff, the off switch, urgent runs and the deadline.
  - Covers that a workflow which did not opt in gets nothing, and that curiosity and compactor-digest still spend an attempt.
  - Checks that every opted-in graph declares its counter.
- `services/orion-durable-runs/tests/test_listener_wait_postgres.py`: 2 new real-runtime tests. A mid-turn loss is re-sent and completes. A run that keeps losing Hub fails after 2 re-sends.
- `services/orion-durable-runs/evals/deploy_window_listener_wait.py`: adds an in-flight window with an opt-in-off control and a poison-request case.
- `services/orion-durable-runs/evals/listener_wait_mutations.py`: 8 new mutants, 2 old strings updated.
- `services/orion-durable-runs/README.md`: documents the in-flight rule.

## Schema / bus / API changes

- Added: a `case` field in the `run.listener_wait` event detail, and `case` in the parked hold dict. Both are internal to durable-runs, and nothing outside it reads them (checked in review).
- Removed / Renamed: none.
- Behavior changed: the in-flight `RpcNoListener` on `reading.turn` now waits and re-sends, at most 2 times.
- Compatibility notes: old checkpoints read the new counter as 0.

## Env/config changes

- Added / removed / renamed keys: none. `DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES=0` still turns off both waits.
- `.env_example` updated: no.
- Local `.env` synced: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-durable-runs: pytest tests (ORION_ADMISSION_TEST_DSN = disposable postgres:16)
  -> 818 passed, 1 skipped
orion/core/bus/tests -> 131 passed
git diff --check -> clean
```

## Evals run

```text
evals/deploy_window_listener_wait.py -> PASS (14/14 checks)
  published-to-nobody: wait off 16 failed; wait on 16 completed, attempt == 1
  in-flight: opt-in off 16 failed (RpcNoListener ... while awaiting its reply); on 16 completed
  poison: 16 failed after exactly 2 re-sends each (3 sends per run)
evals/listener_wait_mutations.py -> 28/28 mutants killed
```

## Docker/build/smoke checks

```text
Not run. No dependencies, ports or startup steps changed. The live behavior is UNVERIFIED until
the next Hub redeploy after this is deployed. To check it: look for run.listener_wait rows whose
detail has case = lost_in_flight on reading-* runs, and confirm those runs end completed.
```

## Review findings fixed

- Finding: the safety comment overstated the claim. A live Hub that is only resubscribing can still be running the original turn, so a re-send can run beside it outside GPU admission.
  - Fix: the comment in `reading_graph.py` and this report now state this cost and its bound.
  - Evidence: `app/reading_graph.py` comment.
- Finding: the eval's `inflight_resends_bounded` check could never fail.
  - Fix: added a poison-request replay. Its check `poison_capped` requires exactly 2 re-sends, then `failed`.
  - Evidence: the eval prints `poison_capped: true`, 3 sends per run.
- Finding: the hold key's value did not say which case fired.
  - Fix: the hold now carries `case`, and the comment says the count is only a truthy marker.
  - Evidence: unit tests assert `hold["case"]`.
- Finding: a future opt-in without a declared state key would never trip its cap.
  - Fix: a docstring requirement, plus `test_every_graph_that_opts_into_in_flight_resends_declares_its_counter`.
  - Evidence: the test passes.

## Restart required

From the clean primary `main` checkout, after merge:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: during a slow resubscribe, a duplicate turn can briefly run beside the original turn, outside GPU admission.
  - Mitigation: at least 60 s backoff, a cap of 2, and the same exposure the seed-queue retry already had.
- Severity: low.
  - Concern: duplicate stance/self-prediction trace rows for a killed turn.
  - Mitigation: they are keyed by a distinct correlation id, so they record a turn that really ran.

## PR link

See the PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
