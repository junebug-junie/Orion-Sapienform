## Summary

- The Hub full-suite CI job took 12-14 min and sometimes hung until the 30 min timeout. This makes it ~2.5x faster and makes hangs fail in 3 min with a stack trace.
- Root cause of the slowness: an autouse fixture in `services/orion-hub/tests/conftest.py` wiped every `scripts.*`/`app.*` module before each of the 4,611 tests. About 200 tests then rebuilt the whole Hub app (`scripts.main`, ~2 s each). It now wipes only when the wrong `scripts`/`app` package is loaded.
- Second root cause, CI only: Hub's startup connects to `ORION_BUS_URL`, whose default is the live tailscale bus. From a GitHub runner that address goes nowhere, so 16 tests each waited ~30 s for the connect to time out (~8 of the CI job's 10.5 min). On circe the same tests joined the **live** bus. `conftest.py` now points tests at a closed local port (`redis://127.0.0.1:1/0`), which refuses at once and takes the same "bus unavailable" path CI always took.
- Two tests (`test_memory_graph_routes.py`, `test_autonomy_goal_actions.py`) relied on that wipe; they now fetch reload targets with `importlib.import_module`.
- Workflow: a newer push to the same PR cancels the older run (main pushes are never cancelled). `pytest-timeout` fails any single test after 180 s and prints every thread's stack.

## Outcome moved

Local full run: 9:04 -> 3:48, same 4,465 passing. CI test step was ~770 s; expect ~5 min. The hung runs (e.g. 41515579895, stuck at 40% for 25 min with no output) will now name the hung test.

## Files changed

- `services/orion-hub/tests/conftest.py`: conditional purge (`_is_hub_package`); `_detach_from_live_bus()`.
- `services/orion-hub/tests/test_memory_graph_routes.py`, `test_autonomy_goal_actions.py`: `import_module` before `reload`.
- `.github/workflows/orion-hub-full-suite.yml`: concurrency, pytest-timeout, `--durations=20`.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest services/orion-hub/tests (before)  4465 passed, 146 skipped in 544.58s
CI run 41600381246 (purge fix only)       test step 644s; 16 tests at exactly ~30s (bus connect)
pytest with ORION_BUS_URL=redis://10.255.255.1 (CI-like dead bus), after bus detach:
                                          1 failed, 4465 passed, 146 skipped in 188.22s; slowest test 7s
pytest services/orion-hub/tests (after)   1 failed, 4465 passed, 146 skipped in 201.69s (after review fixes)
```

The 1 failure, `test_route_engine_reuse.py::test_no_new_per_call_engines_in_shared_orion_code`, fails identically on origin/main: `orion/self_monitor/report.py:collect` (#3041) calls `create_engine` per call, which the guard from #3039 forbids. Out of scope here; left for the self-monitor lane.

## Evals run

None: CI-only change, no eval surface.

## Review findings fixed

- Finding: `_is_hub_package` used `any()`; Hub's `app` is a namespace package, so in a combined multi-service pytest run its `__path__` could include another service's `app` and skip a needed purge.
  - Fix: require every `__path__` entry to be Hub's (`all()`, non-empty).
  - Evidence: full hub suite rerun below.
- Finding: workflow comment implied the run continues after a timeout; `--timeout-method=thread` stops the whole run.
  - Fix: comment reworded.
- Finding: `pre_turn_appraisal_off` docstring said the purge runs every test.
  - Fix: reworded to "can purge".

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: tests that mutate Hub module globals without monkeypatch now share state across tests. Mitigation: full suite passes in CI order; any new leak shows as an order-dependent failure, not a silent pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3059
