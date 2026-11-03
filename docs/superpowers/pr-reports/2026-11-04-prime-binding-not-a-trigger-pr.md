# fix(prime): a change in what binds the body no longer wakes Prime; every thought records why it woke

## Summary
- Prime no longer wakes because a different body limit became the tightest one. That limit still shows in every thought, under BODY. It just stops being a reason to think.
- Each thought's act note now starts with `woke: <what woke it>`, so the reasons Prime wakes can be counted from the ledger instead of guessed.

## Outcome moved
On 11-03, from 09:43 to 21:29 UTC (after attention moves had already stopped waking Prime), the live ledger shows:
- Prime thought 26 times, almost every time right at the 5-minute floor.
- 21 of those 26 thoughts did nothing.
- 4 dropped a worry, and every one gave the reason "lane X is no longer the binding constraint".
- 1 asked a question. It was the only thought that moved anything.

Over 3 days, the set of binding limits changed 25 times a day:
- median spell: 12 min
- spells under 5 min: 29 of 75
- spells under 30 min: 52 of 75

This is the market adjusting its prices. It is not news for a mind.

After the change, a replay on live data (`orion/prime/evals/run_prime_wake_replay.py`) gives:
- 19 thoughts in 90.9 h, about 5 a day.
- 12 of them woken by the 6-hour quiet floor.
- Thoughts on every day.
- No gap longer than 6.0 h.

## What still wakes Prime
- one of its own questions or regulate acts settling;
- a worry fading;
- first wake;
- the 6-hour quiet floor.

## Cost
A genuinely new strain (for example the cabinet heating up) is now seen at Prime's next thought, within 6 hours, not straight away. Heat bound 0 times in the 3 days measured. If that ever matters, a heat-only exception is a small addition.

## Files changed
- `orion/prime/state.py`: the binding set is remembered for the frame but is never a change; docstring has the evidence.
- `orion/prime/thought.py`: new `woke_note` helper.
- `services/orion-mind/app/prime_loop.py`: the thought's close note starts with what woke it.
- `orion/prime/evals/run_prime_wake_replay.py`: docstring states the new result.
- `orion/prime/evals/run_prime_frame_eval.py`: fixture text no longer uses a binding change as the example change.
- Tests:
  - `orion/prime/tests/test_prime_core.py`: binding never wakes Prime; wake note is capped.
  - `services/orion-mind/tests/test_prime_loop.py`: every thought records why it woke.
- `services/orion-mind/README.md`.

## Schema / bus / API changes
None. The persisted state shape is unchanged (`seen_binding` is still written).

## Env/config changes
None.

## Tests run
```text
orion/prime                 77 passed
services/orion-mind/tests  200 passed
orion-static-gates.yml      all steps PASS locally
```

## Evals run
```text
run_prime_wake_replay.py: 3/3 PASS (5.0 thoughts/day; floor woke 12 of 19; longest gap 6.0 h)
run_prime_frame_eval.py: 11/11 PASS
```

## Review findings fixed
- **Only a successful thought got the wake note.**
  - Fix: the reason is captured before the model call, and every close of a thought act carries it: success, preempted by a Juniper turn, unparseable answer, or a crash.
  - Evidence: the preempted and crash tests now assert the note starts with `woke:`.
- **The 500-character note cap could cut off the move.**
  - Fix: the move now comes right after the counts, ahead of the drop reasons and warnings, which have no length limit.
  - Evidence: `test_the_move_comes_before_unbounded_drop_reasons_in_the_note`.
- **A comment implied `seen_binding` feeds the frame.** It doesn't; the frame reads the live body.
  - Fix: the comment and docstring are corrected.
- **Test fixtures still used a binding change as the example pending change.**
  - Fix: they now use the quiet-floor text.

## Restart required
```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns
- Severity: low
  - Concern: Prime may now think too rarely.
  - Mitigation: the fix is to shorten the quiet floor, not to bring the noisy trigger back. Measure it from the new `woke:` notes after a few days.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
