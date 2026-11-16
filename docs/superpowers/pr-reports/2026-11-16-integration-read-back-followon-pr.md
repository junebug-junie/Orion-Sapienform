# Integration patch 3 follow-on: ablation eval, arm-blind curiosity test, Prime block fit fix

Refs #3425 (epic #3434). Patch 3 is merged (#3499 reverie, #3501 Prime, #3503 curiosity). This PR adds only what main
still lacked: a way to measure whether a cite is real, a guard that keeps curiosity's read blind to the running
experiment, and a fix to a Prime block bug that hid one of its own lines.

## Summary

- **Ablation eval.** A new weekly eval checks whether a cited item really shaped what an organ wrote. It runs the
  same prompt twice, once with the cited line and once without it. If the output does not change, the cite is
  "hollow". The offline lane runs in CI on fixtures built from the merged code's own renderers.
- **Curiosity stays blind to the experiment.** A new test runs a real curiosity tick under each of the three offer
  arms. The request to durable-runs and the "other organs" section must be the same, byte for byte, and name no
  arm. A second test refuses any arm, view or propensity name in the read path.
- **Prime block fix.** A tending line could push the "Body" line out of the Prime block without anyone noticing.
  The check that should stop this compared refs only, and the Body line shares its ref with the "Asks" line. The
  check now also counts lines.
- **CI.** #3503's own curiosity tests (`test_curiosity_now.py`, `test_curiosity_event_wants.py`) ran in no
  workflow. They run now, with the new test files.
- **Dashboard check.** Prime's frame no longer reads the curiosity dashboard on main (`_ask`, `prime_loop.py:1025`).
  Only settling still reads it, which the acceptance check allows. Nothing to remove.

## Outcome moved

- Before: no way to tell a real cite from a hollow one. After: a harness that reports the hollow-cite share, and
  proves it can find one (the planted hollow fixture is caught).
- Before: nothing tested that curiosity's now read is the same under every arm. After: a tick-level test and a
  structural test. Both fail when an `arm` parameter is planted in the read (checked by mutation).
- Before: the Prime block dropped its Body line in 135 of 1,440 searched states (two tending lines, room tight).
  After: 0 of 1,440.

## Current architecture

- Orion's now is read by Prime (each thought), reveries (once per chain) and curiosity (each kickoff). Each organ
  shows the items with an `[experience:<id>]` token and posts its cites to `POST /temporal-self/now/use`.
- Cites were counted, but nothing measured whether they were real.
- The Prime block (`orion/prime/block.py`) adds up to 2 tending lines when they fit. The fit check compared the refs
  of the block's own lines before and after making room.

## Architecture touched

- `orion/integration/evals/` (new): eval, fixture builder, fixtures, tests.
- `orion/prime/block.py`: one condition in `render_prime_block`.
- Tests only for curiosity. No service code, schema, bus, env or config change.

## Files changed

- `orion/integration/evals/run_now_ablation_eval.py`: the eval (offline stub lane, live local-model lane).
- `orion/integration/evals/now_ablation_fixtures.py`: builds the 4 fixtures from the merged renderers (Prime
  `build_frame` + `system_prompt`, the real `reverie_narrate.j2` with `render_now_block` lines, curiosity
  `format_now_section`). It loads orion-thought's `app` package under an alias, so another service's `app` never
  stands in for it.
- `orion/integration/evals/fixtures/now_ablation_cases.jsonl`: the 4 cases (one planted hollow cite).
- `orion/integration/evals/test_now_ablation_eval.py`: harness tests plus a drift test (fixtures equal what the
  renderers build today).
- `orion/prime/block.py`: the fit check also requires that the re-render drop no more own lines.
- `orion/prime/tests/test_prime_block_fit.py`: the failing-first test for that bug.
- `services/orion-hub/tests/test_curiosity_now_arm_blind.py`: the arm-blind tick test and the structural test.
- `.github/workflows/orion-static-gates.yml`: eval tests + offline lane in the integration gate step.
- `.github/workflows/temporal-self-readers-tests.yml`: eval tests (incl. the drift test, which needs jinja2 + httpx)
  and path triggers.
- `.github/workflows/orion-reading-tests.yml`: `test_curiosity_now.py`, `test_curiosity_now_arm_blind.py`,
  `test_curiosity_event_wants.py`.

## How the eval decides

- Ablation removes exactly the one line that carries the cited `[experience:<id>]`. A cite whose line is not in the
  prompt is refused, not scored (removing nothing would read as hollow).
- "Changed" means the two outputs differ by at least 0.25 (token-set Jaccard distance), or the full output names
  the cited event id and the ablated one does not. The bare id is matched, because the organs cite in three shapes:
  Prime as `{"kind": "experience", "id": "<id>"}` in `refs`, reverie in `cites`, curiosity as `[experience:<id>]`.
- Offline result on the fixtures: 4 cites, 3 changed, 1 hollow (the planted one). These are harness numbers, not
  Orion's.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: the Prime block shows fewer tending lines when showing
  them would drop one of its own lines. Compatibility notes: none.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
orion/prime/tests/test_prime_block_fit.py on main (before the fix): 1 failed, 1 passed
orion/prime (after): 256 passed
services/orion-mind/tests: 412 passed
orion/prime/evals/run_prime_block_eval.py: PASS, 0 failing checks
services/orion-hub/tests/test_curiosity_now_arm_blind.py: 4 passed
  mutation (plant an arm param in curiosity_now.read_for_kickoff): 2 failed, as intended; file restored
test_curiosity_now.py + test_curiosity_event_wants.py + test_curiosity_now_arm_blind.py
  + test_curiosity_investigation.py: 181 passed
tests/test_check_integration_seam.py + orion/integration + orion/prime + orion/curiosity/tests: 570 passed, 29 skipped
orion/temporal_self/tests + orion/thought/tests + orion/integration/evals: 452 passed
orion-thought test_reverie_orion_now.py + test_reverie_recent_arcs.py: 43 passed
scripts/check_integration_seam.py: OK (16 organs)
minimal venv with the static-gates deps only (pydantic, pydantic-settings, PyYAML, pytest):
  orion/integration/evals 7 passed, 1 skipped (drift test needs jinja2); offline lane PASS
```

## Evals run

```text
python -m orion.integration.evals.run_now_ablation_eval --offline
  cites 4, changed_share 0.75, hollow_share 0.25, hollow = [prime-hollow (planted)], refused []
  PASS offline harness check
Live lane: UNVERIFIED. The local OpenAI-compatible endpoint (127.0.0.1:8222) refused connections from athena.
```

## Live state at the time of this PR (from the coordinator, after #3499/#3501/#3503 deployed)

- Reverie: 17 reads of the now, 22 cites.
- Curiosity: 4 reads, 4 cites. All 4 cite `gpu_wait` experiences, from real investigations.
- Prime: 2 reads, 0 cites.
- Integration rate: 26 of 5,281 experiences used by a different organ.

What this means: reading and citing works for all three organs. Whether the cites are real is exactly what this
eval measures, and it cannot run on live cited runs yet: no organ keeps the prompt it saw (#3516). Curiosity's 4 of
4 cites going to `gpu_wait` is worth a look once the live lane can replay them.

## Docker/build/smoke checks

```text
Not run: no service code, image, compose or config changed. Tests and the offline eval lane only.
```

## Review findings fixed

(filled in after the review subagent)

## Restart required

```text
No restart required. The Prime block fix is in orion/prime, which Hub imports: it takes effect at the next normal
Hub rebuild after merge.
```

## Risks / concerns

- Severity: low. Concern: the drift test skips where jinja2 or httpx is missing. Mitigation: the
  temporal-self-readers job installs both and runs it.
- Severity: low. Concern: the arm-blind tick test patches `draw_offer_arm` and `plan_offer` in the loop's module
  globals (the same pattern as `test_curiosity_investigation.py`). If the draw moves to another module, the patch
  misses and every tick runs one arm. Mitigation: the structural test still guards the read path.
- Severity: medium. Concern: the live ablation number is UNVERIFIED until the organs keep their prompts.
  Mitigation: issue #3516 opened (follow-up, area:self-model, on the Orion project board).
- Issues opened: #3516. Issues closed: none.

## PR link

(filled in after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
