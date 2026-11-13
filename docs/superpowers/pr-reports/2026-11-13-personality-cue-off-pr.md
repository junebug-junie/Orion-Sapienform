## Summary

- Turns the personality cue off in hub (outreach and curiosity prompts) and thought (chat stance and reverie).
- Why: the personality frame held sentences that automated jobs repeat ("prediction error is unusually high", "wrote the answer into the graph first"), not personality. Orion read them back as "my tendencies" and repeated them. The outreach tick sent Juniper one of them 3 times in 2 days.
- Step R0 of personality design rev 3 (#2726, merged 11-13): choices only, held by Prime.
- The reducer keeps running, so no history is lost. Rev 3 R1 replaces it.

## Outcome moved

Orion stops reading job boilerplate as their own character in chat, reverie, outreach and curiosity.

## Files changed

- `services/orion-hub/.env_example`, `services/orion-hub/app/settings.py`: `HUB_PERSONALITY_CUE_ENABLED` defaults to false.
- `services/orion-thought/.env_example`, `services/orion-thought/app/settings.py`: `PERSONALITY_CUE_ENABLED` defaults to false.
- `services/orion-hub/tests/test_personality_cue_reader.py`: the flag test asserts off.
- `services/orion-thought/tests/test_personality_cue.py`: an autouse fixture turns the cue on, because these tests exercise the reader itself.

## Env/config changes

- Changed defaults: `HUB_PERSONALITY_CUE_ENABLED=false`, `PERSONALITY_CUE_ENABLED=false`.
- Local `.env` set to false by hand. The sync script adds keys and does not change values.
- Env parity: PASS.

## Tests run

```text
hub cue tests: 28 passed
thought tests/test_personality_cue.py: 20 passed
check_env_template_parity.py: PASS (89 services)
```

## Restart required

```bash
scripts/rebuild_services_from_git_diff.sh   # after merge: rebuilds orion-hub and orion-thought
```

## Risks / concerns

- Low: the cue no longer reaches these prompts. That is the intent. Rev 3 R3 brings personality back through the Prime block once it is built from choices and Juniper has approved it.
