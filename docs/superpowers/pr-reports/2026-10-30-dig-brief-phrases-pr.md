## Summary

- Orion's always-on dig line now maps Juniper's two rung phrases to commands: "name your organs" → `name-organs`; "claim or write your capacities" → `capacities` (then act on its last line, which holds the ready `capacity claim` command).
- Kept every pointer the tests require (power/cost → `body-economy hosts`, recent PRs → `history 2`, not the GitHub tool). 389 of 400 chars; cap not raised.

## Outcome moved

Live 2026-10-30: Juniper said "Claim your capacities." twice. Orion answered both times with a prose self-portrait and ran no dig command; `/mnt/orion-fcc/self/` still holds only `organ_names.jsonl`. The only place Orion learns dig commands every turn is this line, and it never mentioned capacities or organ naming.

## Current architecture

`orion/dig/brief.py` `DIG_BRIEF_LINE` is appended to every non-reading harness prompt (`orion/harness/prefix.py`). Everything else is behind `dig help`.

## Files changed

- `orion/dig/brief.py`: new wording.
- `orion/dig/tests/test_dig.py`: test that both phrases and both commands are in the line.

## Tests run

```text
pytest -q orion/dig                                75 passed
pytest -q orion/harness -k "prefix or brief"       54 passed
python orion/physiology/evals/physiology_state_skill_eval.py   rc=0, fails []
```

## Evals run

physiology_state_skill_eval (checks the brief's power pointer and cap): pass.

## Review findings fixed

See PR comments; review ran in a subagent.

## Restart required

Code is baked into the harness image; the post-pull rebuild maps `orion/dig` to orion-harness-governor. Fallback:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low. Concern: a prompt line is a cue, not a guarantee; Orion may still answer in prose. Mitigation: verify live by checking `capacity_claims.jsonl` after Juniper asks again.
