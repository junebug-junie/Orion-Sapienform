## Summary

- The fit hop now asks the model to shorten an over-budget self answer to **about 1,000 characters**. It used to ask for "1,200 or fewer".
- 1,200 is still the hard limit. Anything past it is still cut at a sentence.

## Outcome moved

The first live fit hop (run `8f261b9835f4`, 2026-11-03 22:24 UTC):
- It rewrote a 1,759-character answer.
- The rewrite still came back over 1,200 characters, so the sentence trim cut it to 1,106 and the last sentence or two were lost.

Asking for 1,000 leaves room for the model to overshoot without being cut.

## Current architecture

PR #3079 added the fit hop. `build_answer_fit_prompt` asked for "1200 characters or fewer". `accept_fit_text` trims any reply that is still over budget at a sentence.

## Architecture touched

- New `SELF_FIT_TARGET_CHARS = 1000` in `orion/curiosity/self_inquiry_prompt.py`, used only by the fit prompt.

## Files changed

- `orion/curiosity/self_inquiry_prompt.py`: the fit prompt now asks for the ~1,000-character target.
- `orion/curiosity/tests/test_self_answer_fit.py`: a test checks the target sits below the hard limit and appears in the prompt.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
orion/curiosity/tests                                       192 passed, 23 skipped
services/orion-hub tests/test_curiosity_self_answer_fit.py + test_curiosity_self_inquiry.py   60 passed
```

## Evals run

```text
Live check after deploy: Hub log curiosity_self_fit_stored shows reply_chars on the next over-budget run (UNVERIFIED until then).
services/orion-hub/evals/run_self_answer_fit_eval.py --since 2026-09-01 -> at_cap_mid_word 0, all_time 0, exit 0
```

## Docker/build/smoke checks

```text
Prompt-only change; Hub picks it up on rebuild.
```

## Review findings fixed

- Code review (subagent): no material findings. Checked: no other asserts on the old wording; the only caller is unaffected; the 1,000 target is far above the 300-char FIT_MIN_CHARS floor.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: the model may now cut more than it needs to. Mitigation: the full answer stays on the graph, and 1,000 characters is still far above the 300-character minimum a rewrite must meet.

## PR link

See the PR.
