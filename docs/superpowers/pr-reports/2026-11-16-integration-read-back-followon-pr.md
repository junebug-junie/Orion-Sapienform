# Integration patch 3 follow-on: ablation eval, arm-blind curiosity test, Prime block fit fix

Refs #3425 (epic #3434). Patch 3 is merged (#3499 reverie, #3501 Prime, #3503 curiosity). This PR adds only what main
still lacked: a way to measure whether a cite is real, a guard that keeps curiosity's read blind to the running
experiment, and a fix to a Prime block bug that hid one of its own lines.

## Summary

- **Ablation eval.** A new eval checks whether a cited item really shaped what an organ wrote. It runs the same
  prompt twice, once with the cited line and once without it. If the content of the output does not change, the
  cite is "hollow". The offline lane runs in CI on fixtures built from the merged code's own renderers. It is meant
  to run weekly by hand; nothing schedules it yet (#3516).
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
- "Changed" means the CONTENT of the two outputs differs by at least 0.25 (token-set Jaccard distance). The cited
  id is stripped from both outputs first, in all three shapes the organs cite in: Prime's
  `{"kind": "experience", "id": "<id>"}` in `refs`, reverie's `cites`, curiosity's `[experience:<id>]`. Dropping
  only the cite token never counts: the ablated prompt cannot show the id, so that difference is there by
  construction.
- An empty output is refused, not scored. Two empty outputs would read as "no change", which is a fake hollow.
- In the live lane the full prompt is replayed twice. The difference between those two replays is the noise floor,
  and a cite must beat the noise by 0.10 to count as changed.
- Offline result on the fixtures: 4 cites, 3 changed, 1 hollow (the planted one). These are harness numbers, not
  Orion's. A "parrot" model that repeats every shown id and writes nothing about them scores 0% changed (it scored
  100% before the review fix).

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
after the review fixes:
orion/integration/evals + orion/prime + tests/test_check_integration_seam.py: 285 passed
Hub: test_curiosity_now_arm_blind + test_curiosity_now + test_curiosity_event_wants
  + test_curiosity_investigation: 181 passed
orion/temporal_self/tests + orion/thought/tests + orion/integration/evals: 456 passed
static-gates packages + jinja2 httpx python-dotenv: orion/integration/evals 12 passed, 0 skipped
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

A code-review subagent read the full diff. It confirmed the block test reproduces main's bug, the arm pin works, and
every CI step imports with its job's packages. It found these, all fixed:

- Finding (must-fix): the eval counted "the full output names the cite, the ablated one does not" as change. That
  is true by construction, so a model that only repeats ids read 100% changed, the planted hollow case included.
  - Fix: change is content only; the cited id is stripped from both outputs in all three cite shapes first.
  - Evidence: the parrot model now reads 0.0 changed; `test_only_the_cite_token_differing_is_hollow_in_every_shape_an_organ_cites_in`.
- Finding (must-fix): an empty output (a thinking model that used its whole budget) read distance 0, so hollow.
  - Fix: empty outputs are refused, not scored; the live call sends `enable_thinking: false` and 800 tokens.
  - Evidence: `test_an_empty_output_is_refused_never_scored_hollow`.
- Finding (should-fix): no noise floor in live mode (a local server at temperature 0 still drifts).
  - Fix: live replays the full prompt twice and needs `max(0.25, noise + 0.10)`; noise is in the summary.
  - Evidence: `test_live_noise_raises_the_bar`.
- Finding (should-fix): the offline check and unit tests never exercised the distance path (`--threshold 2.0`
  still passed).
  - Fix: the stub writes labels only; new tests for content-change and token-only cases.
  - Evidence: `--offline --threshold 2.0` now FAILs (asserted in a test). The curiosity fixture's stub now uses
    only the cited item, as a real run takes up one candidate.
- Finding (should-fix): the arm test could pass hollow if `draw_offer_arm` stopped being a module global.
  - Fix: assert both names exist before patching; spy on `_record_offer` and assert the recorded arm is the
    pinned one; assert the replica planner ran on the replica arm only.
  - Evidence: `test_curiosity_now_arm_blind.py` 4 passed.
- Finding (should-fix): the drift test ran only in a path-filtered job, and skipped in static gates.
  - Fix: static gates install jinja2, httpx and python-dotenv; the readers job's PR and push filters cover
    `orion/prime/**`, `orion/integration/**` and `now_section.py`.
  - Evidence: in a venv with exactly the static-gates packages, `orion/integration/evals` runs 12 passed, 0 skipped.
    I ran every static-gates test file with the new packages: 697 passed, 2 skipped. One test failed only in a
    single combined process; it passes alone (46 passed) and in its own CI step (86 passed).
- Finding (nit): fixtures rendered the reverie template with a plain jinja Environment.
  - Fix: `prompt_environment()`, the one production renders with. The fixtures did not change.
- Finding (nit): importing orion-thought's settings loads its `.env` into the process. Fix: said so in the docstring.
- Finding (nit): a block test asserted the code's own formula. Fix: replaced with concrete numbers for the
  reproducing case (1 tending line shown, 1 dropped, body line kept).
- Finding (nit): `eid in full` matched `a:1` inside `a:10`. Fix: whole-token match (it may follow `experience:`).
  The tokenizer also stopped counting "warm." and "warm" as different words.
- Finding (nit): "weekly" with nothing scheduling it. Fix: the docstring and this report say "run by hand" and point
  at #3516.

## Restart required

```text
No restart required. The Prime block fix is in orion/prime, which Hub imports: it takes effect at the next normal
Hub rebuild after merge.
```

## Risks / concerns

- Severity: low. Concern: the drift test skips where jinja2 or httpx is missing. Mitigation: static gates (every
  PR) and the temporal-self-readers job now install both and run it.
- Severity: low. Concern: the arm-blind tick test patches `draw_offer_arm` and `plan_offer` in the loop's module
  globals (the same pattern as `test_curiosity_investigation.py`). Mitigation: the test fails, not passes, if a
  name is gone or the offer row records a different arm than the pinned one.
- Severity: medium. Concern: the live ablation number is UNVERIFIED until the organs keep their prompts.
  Mitigation: issue #3516 opened (follow-up, area:self-model, on the Orion project board).
- Issues opened: #3516. Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3518

🤖 Generated with [Claude Code](https://claude.com/claude-code)
