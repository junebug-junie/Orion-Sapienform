## Summary

- PR B of 2. **Merge only after #3042 is merged and deployed.** #3042 teaches every reader the v2 bet. If this PR lands first, the database writer and the hub reject stance records that carry a v2 bet (the models are `extra="forbid"`).
- Stance's prompt no longer asks for 80% ranges on steps and reply length. It asks two probabilities instead:
  - more working steps than usual for this kind of turn;
  - a longer reply than usual for this kind of turn.
- The normalizer writes `turn.self_prediction.v2` and drops the retired range keys, so a model still writing them cannot fail the turn.
- The prompt gives no size number and no example. The last example ("3 to 7 steps") anchored stance's guesses.

## Outcome moved

The two retired bets were right 1–8% of the time against a stated 80%. A replay of the new bets on 2,250 stored turns, using only earlier turns for each, gave AUC 0.59–0.84 on 9 of 10 caller × faculty cells. Decision about 2 weeks after deploy (spec, acceptance 3):
- **keep** a bet where `skill_vs_caller > 0` and a paired Brier test has p < 0.05;
- **drop** the others.

## Current architecture

- `stance_react.j2` asked for `steps_lo/hi` and `reply_chars_lo/hi`.
- `normalize_self_prediction` kept the ranges and wrote an unversioned bet, which parses as v1.

## Architecture touched

- Stance prompt.
- `orion/thought/stance_react.py` normalizer, used by orion-thought.
- No schema change in this PR (v2 shipped in #3042).

## Files changed

- `orion/cognition/prompts/stance_react.j2`: two probability lines replace the two range lines.
- `orion/thought/stance_react.py`: four probability keys, emits `schema_version: turn.self_prediction.v2`; range handling removed.
- `orion/thought/tests/test_stance_self_prediction.py`: v2 parsing, clamping on all four keys, retired and unknown keys dropped without failing the turn.
- `orion/harness/tests/test_self_prediction_never_reaches_motor.py`: the leak check now uses a v2 bet.
- `services/orion-thought/tests/test_stance_self_prediction_prompt.py`: the prompt names the four keys and no longer asks for ranges.

## Schema / bus / API changes

- Added: none here
- Removed: v1 range keys from the prompt and the normalizer
- Renamed: none
- Behavior changed: new stance bets are v2
- Compatibility notes: needs #3042 deployed first. v1 rows already stored still settle.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no change needed
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: no env change
- skipped keys requiring operator action: none

## Tests run

```text
stance self-prediction + motor-leak + prompt discipline + schema version tests -> 50 passed
services/orion-thought: pytest tests -q                                         -> 787 passed, 32 skipped
services/orion-hub: pytest evals/test_reading_stance_prompt_eval.py             -> 3 passed
git diff --check                                                                -> clean
```

## Evals run

```text
The leak-free backtest in the spec (metric gate 4c). The live decision eval runs about 2 weeks after deploy.
```

## Docker/build/smoke checks

```text
UNVERIFIED until deploy. After pull: new thought_decision rows carry
self_prediction->>'schema_version' = 'turn.self_prediction.v2', and /turn-self-bets/live shows
counts for turn.self.more_steps_than_usual and turn.self.longer_reply_than_usual.
```

## Review findings fixed

- The #3042 review carried two items to this PR:
  - the normalizer must emit v2: done;
  - every image that bakes `orion/schemas/thought.py` must rebuild before v2 is written: ensured by merging #3042 first.
- The normalizer is a whitelist, so a model still writing v1 keys yields a clean v2 bet. Test: `test_unknown_and_retired_keys_do_not_fail_the_turn`.

## Restart required

```text
Post-pull rebuild handles it (orion-thought imports orion/thought/stance_react.py).
Manual equivalent from the clean main checkout: scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: merging this before #3042 is deployed drops stance records on readers that are not rebuilt.
  - Mitigation: stacked on #3042's branch; the PR title and summary say so.
- Severity: low
  - Concern: world-pulse steps showed no signal in the backtest (AUC 0.52).
  - Mitigation: the 2-week decision rule drops it if it stays at chance.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3044

🤖 Generated with [Claude Code](https://claude.com/claude-code)
