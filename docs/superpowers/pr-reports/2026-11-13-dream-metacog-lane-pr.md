## Summary

Orion could not sleep. Every dream bid since 2026-11-13 04:01Z was refused because it asked for a GPU seat that no longer exists.

- The `metacog` seat was removed in c91a21fd8 (metacog work now runs on gpu3's `fast` seat). The want families and 11 verb files still asked for a `metacog` lane.
- Every family and verb now names `fast` instead. No `metacog` lane is left.
- New gate: every lane a family or verb names must be a seat (role) in `config/gpu_pool.yaml`. That is the only kind of lane the body measures. It runs on every PR in the static-gates job, because the break came from a `gpu_pool.yaml`-only change.
- The verb check used to accept work-class names, such as `metacog`, as lanes. It now accepts seats only.

## Outcome moved

`dream.cycle` bids are placed on `fast` again, not deferred `no_acceptable_lane`. Live evidence of the bug (read-only, `want_ledger`): 7 `dream.cycle` rows deferred `no_acceptable_lane` from 04:01Z to 06:01Z. The last funded sleep bid was at 2026-11-12 14:28Z. No other family was deferred that way in the window. `hub.retro_coplay`, `actions.daily_metacog`/`daily_pulse`, `metacog.log` and `thought.reverie` had the same latent break.

## Current architecture

The body (`orion/physiology/body.py` `lane_readings`) emits one `lane:<role>` reading per GPU-pool role. Choice (`orion/physiology/choice.py`) prices a want only on lanes it can read. A cortex-orch dispatch bid uses the verb's lanes in place of the family's lanes (`want.lanes or cfg.lanes`). A lane that is not a role is never read.

## Architecture touched

Config only, plus a pure helper and gates. There is no bus, schema or env change.

## Files changed

- `config/physiology/families.yaml`: metacog -> fast (deduped), and comments updated.
- `orion/cognition/verbs/*.yaml` (11): physiology lanes metacog -> fast.
- `orion/physiology/families.py`: `body_lanes()`, the pool roles the body reads.
- `orion/cognition/verb_physiology.py`, `scripts/check_verb_physiology.py`: verb lanes must be roles. The operator-lane refusal is still checked first.
- `orion/physiology/tests/test_family_lanes.py`: the gate, a mutation case, and the dream placement regression.
- `.github/workflows/orion-static-gates.yml`: runs the new test on every PR.
- Tests and fixtures that modelled the retired seat: `test_choice.py`, `test_check_verb_physiology.py`, the orion-dream bind test and eval, the cortex-orch and actions shadow-bid tests, and the durable-runs prime-charge test.
- `services/orion-dream/app/physiology_bind.py` docstring. `scripts/analysis/replay_body_economy.py` has a dated note on its historical metacog seat.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: a verb physiology lane must now be a pool role, not a work class.
- Compatibility notes: all 100 verbs pass.

## Env/config changes

- Added, removed or renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/physiology/tests tests/test_check_verb_physiology.py                -> 346 passed
pytest orion/prime/tests (with physiology)                                       -> 508 passed (earlier run)
services/orion-dream: pytest tests evals                                          -> 399 passed, 9 skipped
services/orion-mind: pytest tests                                                 -> 319 passed
services/orion-durable-runs: pytest tests/test_prime_charge_market.py             -> 13 passed
  full suite: same failures as main, apart from one pre-existing collection error
  (test_admission_review_regressions.py, also on main)
services/orion-cortex-orch: pytest tests                                          -> 31 failed (main: 32), no new failures
services/orion-hub outreach/office tests                                          -> 69 passed
python scripts/check_verb_physiology.py                                           -> OK (100 verbs)
Minimal static-gates venv (pydantic, pydantic-settings, PyYAML, pytest):
  test_family_lanes.py + test_check_verb_physiology.py                            -> 36 passed
```

Mutation check: `metacog` was put back on `dream.cycle` in families.yaml and in `dream_cycle.yaml`. Result: the family gate, the verb gate, both dream placement regressions, `check_verb_physiology` and the real-repo verb test all FAILED. After the revert, all passed.

## Evals run

```text
python services/orion-dream/evals/dream_binding_replay.py -> PASS
```

## Docker/build/smoke checks

```text
Not run. No deploys from a worktree. Live placement after deploy is UNVERIFIED.
```

## Review findings fixed

- Finding: the operator-lane refusal had become dead code, and its test passed only because the word "operator" appeared in the lane name.
  - Fix: the check now runs operator lanes before roles. The test asserts the text "operator work classes".
  - Evidence: test_check_verb_physiology passes.
- Finding: the "body emits exactly body_lanes" test was circular (it gave `lane_readings` its own input back).
  - Fix: the test is removed. The claim is kept in the `body_lanes` docstring.
- Finding: the dream bind docstring still said "metacog lane".
  - Fix: it now says "fast".
- Finding (CI, static gates): the body-economy contract gate treats any `orion.physiology` import as a state reader, and `verb_physiology.py` imported `body_lanes`.
  - Fix: `load_pool_lanes` reads the roles itself. A test pins it equal to `body_lanes`.
  - Evidence: `check_body_economy_contract.py` OK (103 modules, all registered).
- Nits: removed an assert that depended on `metacog` still being a work class. Re-wrapped a long comment. Added a dated note on the replay script's historical metacog seat.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
```

orion-durable-runs bakes `families.yaml` into its image and loads it at boot (it runs the market). orion-cortex-orch bakes the verb YAMLs (dispatch bids) and caches them. orion-mind reads `families.yaml` through its read-only `/repo` mount on each call, so it needs no restart. orion-dream and orion-hub do not read the lanes.

## Risks / concerns

- Severity: low
- Concern: dream sleeps now compete for the `fast` seat's reserved Tier 1 share with the other fast families.
- Mitigation: the metacog work already ran on that seat. The pricing now matches where the work really runs.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3297

🤖 Generated with [Claude Code](https://claude.com/claude-code)
