# fix(substrate): attention salience floor default 0.2 -> 0.05 everywhere

## Summary

- The rule for how important a substrate node has to be before it can compete for Orion's attention had two different defaults. The operator template said 0.05; the code said 0.2. Every code default is now 0.05.
- 0.05 was the deliberate, later decision: commit `263b762d6` (2026-09-05) lowered it after measuring that no live node cleared 0.2. That commit only changed `.env_example`. The module constant, `settings.py` and the compose fallback were never updated.
- Adds a static test that reads all four places the default lives and fails if any of them differ. It runs in `orion-static-gates.yml`.
- Moves one test fixture: its "calm" node sat at exactly 0.05, which used to be below the floor and now meets it.
- Spec: `docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md` (PR #2528), "Real bugs found" item 6.

## Outcome moved

Production behavior does not change. The live container already reads 0.05 from the service `.env`. What changes is every run that does **not** have that `.env` value: tests, a fresh host, or a dropped key. Those runs used to fall back to 0.2 without saying so. On today's graph that drops 1 of the 3 nodes that compete. In the 2026-09-05 measurement it cut the share of ticks with two or more competitors from 36.1% to 6.6%. The drift can no longer recur unnoticed.

Live evidence (read-only, 2026-10-10 ~01:47Z):

- `docker exec orion-athena-substrate-runtime printenv ORION_ATTENTION_BROADCAST_MIN_SALIENCE` -> `0.05`. The primary `services/orion-substrate-runtime/.env:93` is also `0.05`.
- Live graph snapshot: inside the container, `build_substrate_store_from_env().snapshot()` was run through the same `is_cognitive_node` + `_node_salience` path that the broadcast uses. Result: 5,400 nodes, and only 5 cognitive nodes have any salience: 0.3567 (`harness_closure`), 0.204 (`chat`), 0.075 (`perception`), 0.0387 (`bus_synaptic`), 0.0246 (`biometrics`). **3 clear 0.05 and 2 clear 0.2.** At 0.2, `perception` would be dropped, and `chat` would only just clear the floor (0.204).
- `substrate_attention_broadcast_log`, last 50 ticks (01:12-01:47Z), all recorded with `min_salience: 0.05`. Admitted signal count per tick: 1 signal on 14 ticks, 2 on 20, 3 on 13, 4 on 2, 5 on 1. The log does not store per-signal salience. A lower bound (`evidence_strength / loop confidence`, which is at most the signal's salience) puts 60 of 106 open loops at >= 0.2 for certain. The other 46 have a lower bound below 0.2, so **up to 43%** of admitted loops would have been dropped at 0.2.

## Current architecture

`orion-substrate-runtime`'s `_attention_broadcast_tick` (worker.py) passes `settings.attention_broadcast_min_salience` into `orion/substrate/attention_broadcast.py::build_substrate_attention_frame`. That function calls `substrate_pressure_signals`, which drops any node whose `dynamic_pressure` is below the floor. Before this patch the default lived in 4 places: module `DEFAULT_MIN_SALIENCE = 0.2`, settings `Field(0.2, ...)`, compose `${...:-0.2}`, and `.env_example` `0.05`.

## Architecture touched

Defaults only. No new key, no change to the schema or the bus, no change to worker.py.

## Files changed

- `orion/substrate/attention_broadcast.py`: `DEFAULT_MIN_SALIENCE` 0.2 -> 0.05, plus a comment that points to the parity test.
- `services/orion-substrate-runtime/app/settings.py`: Field default 0.2 -> 0.05.
- `services/orion-substrate-runtime/docker-compose.yml`: fallback `:-0.2` -> `:-0.05`.
- `tests/test_attention_broadcast_min_salience_parity.py`: new static parity gate. It uses only AST/regex parsing, so it imports no service code.
- `.github/workflows/orion-static-gates.yml`: runs the parity gate.
- `orion/substrate/tests/test_attention_broadcast.py`: the "calm" fixture goes from 0.05 to `DEFAULT_MIN_SALIENCE / 5`, so it always stays below the floor.
- `docs/superpowers/pr-reports/2026-10-10-attention-broadcast-floor-drift-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the default floor is now 0.05 when the env var is unset.
- Compatibility notes: production already sets 0.05, so nothing changes there.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no. It was already 0.05, which is the value everything else now matches.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, because no template changed. The primary `.env` is already 0.05.
- skipped keys requiring operator action: none

## Tests run

```text
tests/test_attention_broadcast_min_salience_parity.py                 1 passed
Mutation check: reset each surface to 0.2 one at a time, ran the test, then restored it
  attention_broadcast.py -> 1 failed | settings.py -> 1 failed
  docker-compose.yml     -> 1 failed | .env_example -> 1 failed

orion/substrate/tests + tests/test_voluntary_attention_wiring.py + tests/test_cognitive_substrate_phase4_dynamics.py
  branch: 5 failed, 1066 passed, 36 skipped
  origin/main baseline: 5 failed, 1065 passed (same 5 failures: test_felt_state_self_definition_lane x3,
  test_cognitive_substrate_phase4_dynamics x2. These were already failing before this patch.)
  Before the fixture change, test_high_pressure_node_wins_over_calm failed because its calm node sat exactly at 0.05.

services/orion-substrate-runtime/tests (--continue-on-collection-errors)
  14 failed / 11 errors, and the failure set is IDENTICAL to the origin/main baseline. These are environment
  failures that were already there, unrelated to this patch.

scripts/check_env_template_parity.py      PASS (94 services)
scripts/check_env_key_single_source.py    OK
scripts/check_compose_no_relative_mounts.py PASS
scripts/check_service_env_compose_parity.py orion-substrate-runtime: 17 keys missing, already
  missing before this patch; this key is not one of them.
```

## Evals run

```text
No eval harness covers this default. The parity test is the deterministic gate for it, and the live
salience distribution above is the measurement behind the value.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-substrate-runtime build  -> rc=0, image built (not deployed)
```

## Review findings fixed

The review subagent ran against `git diff origin/main...HEAD` and found no material issues. It confirmed:

- The only production caller, worker.py, always passes the floor value explicitly.
- Hub and cortex-exec never call these functions.
- `check_metric_lineage --gate` and `check_definition_drift --gate` both pass, and no metric definition changed.

Two nits were fixed:

- Finding: the settings parser ran `ast.literal_eval` on every Field alias. If some unrelated field had a non-literal alias (for example `AliasChoices`), the gate would go red for the wrong reason.
  - Fix: the parser now matches only `ast.Constant` aliases.
  - Evidence: the parity test passes, and resetting settings.py to 0.2 still fails it.
- Finding: the "calm" fixture hardcoded 0.01. If the floor ever dropped below 0.01, the test would quietly stop proving "below the floor is excluded".
  - Fix: the fixture is now `DEFAULT_MIN_SALIENCE / 5`.
  - Evidence: `test_attention_broadcast.py`: 13 passed.

Accepted as is: the parsers return the first match, and the local `.env` is out of CI scope because it is gitignored.

## Restart required

```text
No runtime change: the live container already runs 0.05 from .env. After merge, deploy from the
primary checkout on main only to pick up the code:
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
- Concern: anything that relied on the 0.2 default, such as a caller of `build_substrate_attention_frame` that does not pass `min_salience`, now admits more nodes.
- Mitigation: worker.py always passes the setting explicitly, and production was already at 0.05.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2573

🤖 Generated with [Claude Code](https://claude.com/claude-code)
