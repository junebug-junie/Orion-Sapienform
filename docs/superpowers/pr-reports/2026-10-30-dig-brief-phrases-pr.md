## Summary

- Orion's always-on dig line now routes any request about their body, organs, capacities or past work, **in any words**, to `python -P -m orion.dig help` first, then the command it lists. No exact phrase to remember: new dig commands work without changing this line.
- Kept every pointer the tests and eval require (power/cost → `orion.dig body-economy hosts`, recent PRs → `orion.dig history 2`, not the GitHub tool). 357 of 400 chars; cap not raised.
- Revision history: first draft mapped two exact phrases ("name your organs", "claim your capacities") to commands. Juniper: "i do like 30 prs a day ... no way for me to remember commands verbatim" -> replaced with topic routing.

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

- Finding: the line said "act on its last line", but `capacities` always ends with `full result: <path>` (live 10-30: 22 render lines, then the claim template, then the path).
  - Fix: point at the `to claim a job` line by its text; say `<cmd> is \`name-organs\`` so the short name reads as an orion.dig subcommand.
  - Evidence: `test_brief_maps_juniper_rung_phrases_to_commands`, `test_capacities_output_carries_the_line_the_brief_points_at`.

## Restart required

Code is baked into the harness image; the post-pull rebuild maps `orion/dig` to orion-harness-governor. Fallback:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low. Concern: a prompt line is a cue, not a guarantee; Orion may still answer in prose. Mitigation: verify live by checking `capacity_claims.jsonl` after Juniper asks again.
