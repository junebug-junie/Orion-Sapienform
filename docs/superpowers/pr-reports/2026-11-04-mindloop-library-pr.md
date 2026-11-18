## Summary

This is patch 1 of the curiosity replica (design PR #3093, section 10). Orion Prime's background thinking loop
now runs on a shared loop library, so a second small mind can use the same machinery without copying it. Prime
behaves exactly as before.

- New package `orion/mindloop/` holds the parts any resident loop needs: the tick timer, start and stop with bus
  retry, yielding to Juniper's turn, the self-monitor check before a thought, the bid that must be funded, the
  model call (seat wait, "never reached the model" handling and backoff, the starved route, one retry on a
  server error), opening and closing the thought act, and saving state under the loop's own Redis key.
- Prime is the library's first instance (`PrimeRuntime(MindLoopRuntime)`). `prime_loop.py` went from 998 to
  about 640 lines and keeps only what Prime does: its reads, concerns, frame, answer, its two moves, and how
  they settle.
- A tiny test instance ("echo") runs on the same library with its own key, family, actor and settings. It
  proves a second instance works, including next to Prime on one bus and one Redis.
- Proof of no behavior change: a recorded replay of 17 tick sequences, captured from the old code before the
  move, gives the same acts, bids, model calls, saved state, log lines and the same order of all of them on the
  new code. The live
  Prime state captured from Redis today loads and saves back byte for byte.

## Outcome moved

The curiosity replica (patch 3) can now be written as about 100 lines of its own logic instead of a second copy
of Prime's 600 lines of bid, retry and preemption code. No runtime behavior moved.

## Current architecture

`services/orion-mind/app/prime_loop.py` held everything in one class: generic loop machinery mixed with Prime's
concerns, frame and moves. The no-model backoff lived in `orion/prime/state.py`.

## Architecture touched

- `orion/mindloop/runtime.py`: `MindLoopRuntime`, the shared tick, bid, think and close logic. An instance
  subclasses it, sets `identity` and `state_model`, and implements a short list of hooks (`_upkeep`,
  `_wants_to_think`, `_ask`, `_parse_answer`, `_conclude`, ...). All I/O (bus, model, bid, release, clock,
  Juniper's turns) is injected.
- `orion/mindloop/instance.py`: `LoopIdentity` (name, family, actor, producer, Redis key, settings prefix,
  tier, source, thought component, temperature).
- `orion/mindloop/backoff.py`: `classify_try`, `no_model_backoff`, `note_no_model` and friends, moved
  unchanged from `orion/prime/state.py` (which re-exports them, so old imports keep working).
- `orion/mindloop/gate.py`: `build_act_gate`, the ledger read and the refusal wording, moved from `prime_loop.py`.
- `orion/mindloop/turns.py`: `JuniperTurns` and the one process-wide `JUNIPER_TURNS`, so every instance yields
  to the same Juniper turn.
- Small hooks the replica will need, which do nothing for Prime: `wake()` (end the wait between ticks early,
  when a request arrives), `_want_extras()` (more bid fields), `_on_not_funded()` and `_on_no_model()` (so an
  instance can tell its requester why no thought happened). The runtime also refuses to build an instance whose
  state model lacks the five loop-state fields.
- `config/physiology/contract.yaml`: the bid moved, so the library is now registered as a bidder.
- Deviation 1 from the design text: the design sketched `MindLoopRuntime(instance, ...)` (composition).
  This patch uses a subclass instead, because the existing tests (and the gate code) reach into
  `rt.state`, `rt.gate`, `rt._flush_concern_acts` directly. A subclass keeps every one of them unchanged.
  Prime also stays in `app/prime_loop.py` instead of a new `orion/prime/instance.py`: its I/O is the
  instance, and moving the file would only rename things.
- Deviation 2: the design says every Prime test passes "unchanged". Two test files changed, but only where
  they replace a function: `time.monotonic` is now read in the library, and the gate's view and ledger reads
  live in `orion/mindloop/gate.py`. No assertion changed. A replacement left on the old module would do nothing
  silently, so moving it was required.

## Files changed

- `orion/mindloop/__init__.py`, `runtime.py`, `instance.py`, `backoff.py`, `gate.py`, `turns.py`: new library.
- `orion/mindloop/tests/echo_instance.py`, `test_second_instance.py`: the second instance and its tests,
  plus a check that the library imports nothing from `orion.prime` or `orion.curiosity`.
- `services/orion-mind/app/prime_loop.py`: Prime as instance #1; only Prime-specific code remains.
- `services/orion-mind/app/main.py`: imports `JUNIPER_TURNS` from the library.
- `orion/prime/state.py`: backoff code moved out, re-exported.
- `services/orion-mind/tests/prime_replay_scenarios.py`, `test_prime_replay.py`, `fixtures/prime_replay_golden.json`,
  `fixtures/prime_state_live_2026-11-04.json`: the replay and live-state proof (committed first, on the old code).
- `services/orion-mind/tests/test_prime_loop.py`, `test_prime_self_monitor_gate.py`: monkeypatch targets
  follow the moved code (`time.monotonic` is read in the library; the gate builder's view and ledger reads
  live in `orion/mindloop/gate.py`). No assertion changed.
- `config/physiology/contract.yaml`: registers `orion/mindloop/runtime.py` as a bidder.
- `config/gpu_pool.yaml`, `orion/self_monitor/replay.py`: path comments.
- `.github/workflows/orion-prime-tests.yml`: runs on `orion/mindloop/**` and runs its tests.
- `services/orion-mind/README.md`, `orion/prime/__init__.py`, `orion/prime/evals/run_prime_frame_eval.py`: pointers.

## Schema / bus / API changes

- Added: none on the bus. New Python package `orion/mindloop/`.
- Removed: none.
- Renamed: none. `app.prime_loop` still exports `STATE_KEY`, `JuniperTurns`, `JUNIPER_TURNS`, `PrimeRuntime`,
  `build_act_gate`.
- Behavior changed: none.
- Compatibility notes: Prime's actor (`prime`), act ids, family (`prime`), Redis key (`orion:prime:state`),
  env keys (`PRIME_*`), log line names and persisted state format are unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
orion-prime-tests.yml, every step run locally:
  Prime pure logic (orion/prime)                         OK  77 passed
  Loop library (orion/mindloop)                          OK  9 passed
  References contract                                    OK
  Frame eval on live inputs                              OK
  Wake replay on the live trigger timeline               OK
  orion-mind tests (incl. replay + live state)           OK  203 passed
orion-static-gates.yml, every step run locally:           OK  40 of 40 steps (incl. body-economy contract)
body-atlas-tests, body-economy-analysis, mind-tier-by-caller-tests: all steps OK
orion-durable-runs-tests, orion-sql-writer-tests: every non-Postgres step OK; the Postgres-backed steps
  cannot run here (no service container on this host). This PR's only file on their path list is
  config/physiology/contract.yaml.
Replay proof: prime_replay_scenarios.py run on the pre-refactor tree (b4ae47c) and on HEAD -> identical JSON.
Mutation checks: swapping release/_after_think, dropping the save after a gate refusal, renaming a log line,
  changing the temperature -> each fails test_prime_replay.py.
Import smoke in the real orion-mind image (worktree mounted read-only): app.main imports; one JUNIPER_TURNS;
  PrimeRuntime key orion:prime:state.
```

## Evals run

```text
python orion/prime/evals/run_prime_frame_eval.py   -> all PASS
python orion/prime/evals/run_prime_wake_replay.py  -> all PASS
The no-behavior-change replay (services/orion-mind/tests/test_prime_replay.py) is the eval for this patch.
```

## Docker/build/smoke checks

```text
Not run: no dependency, Dockerfile or compose change. orion-mind's image copies the repo's orion/ package,
so orion/mindloop/ ships with the next rebuild.
```

## Review findings fixed

A review subagent compared the new code with the old one line by line, re-ran the replay on the old tree, and
ran mutations. It found no behavior change. Fixed:

- Finding: the body-economy contract gate failed, because the bid moved into an unregistered module.
  - Fix: `orion/mindloop/runtime.py` registered as a bidder in `config/physiology/contract.yaml`.
  - Evidence: `check_body_economy_contract.py` -> OK (97 modules, all registered).
- Finding: the replay could not see the order between channels or how often state is saved, and it never
  ran with the self-monitor gate on.
  - Fix: one ordered timeline of every publish, save, bid, release and model call, plus 3 gate-on scenarios
    (thought refused, probe spent, concern bet refused). Golden re-recorded on the old tree.
  - Evidence: the old and new trees give identical output. The two mutations the reviewer found unseen now fail.
- Finding: the library had no way for the replica to wake on a request, add bid fields, or learn why
  no thought happened.
  - Fix: `wake()`, `_want_extras()`, `_on_not_funded()`, `_on_no_model()`, all no-ops for Prime, each tested
    on the echo instance.
- Finding: a state model missing a loop field failed only on its first no-model try.
  - Fix: checked when the instance is built (`LOOP_STATE_FIELDS`), with a test.
- Finding: the design deviations were not written down.
  - Fix: recorded above under "Architecture touched".
- Nits fixed: the docstring named a class that does not exist; the "raises NotImplementedError" wording;
  the import test now resolves relative imports and bans `app.*`; the echo instance uses `identity.verb`;
  stale path comments.
- Not changed: the live state fixture contains Orion's real concern text, including the energy budget
  figures. Kept on purpose: the brief said to scrub nothing beyond what is already there, and there is no
  privacy boundary between Juniper and Orion. Juniper can ask for the free text to be blanked; the
  round-trip test would still mean the same.

## Restart required

```bash
# After merge, from the clean primary main checkout synced with origin/main
# (the post-pull rebuild does this on its own when services/orion-mind changes):
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low
- Concern: a refactor of a live loop can change behavior in a path no test drives.
- Mitigation: the replay covers funded, held back, deferred, no bid answer, unparseable, no seat x4 then the
  starved route, 5xx retry, two 5xx, no reply, preempted, Juniper yield, model client raising, a regulate signal
  through its settle, a question settling across a restart, and three self-monitor gate-on paths. All 203 existing orion-mind tests pass with no
  assertion changed. The live loop after deploy is UNVERIFIED until the first `prime_started` and
  `prime_thought` lines appear in the orion-mind log.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3103

🤖 Generated with [Claude Code](https://claude.com/claude-code)
