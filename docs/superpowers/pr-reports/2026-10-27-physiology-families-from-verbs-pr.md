# chore(physiology): families for every tier 1/2 cortex verb (Patch C3 follow-up)

## Summary

- Adds the 15 tier 1/2 verb families that were missing from `config/physiology/families.yaml`: self_study, concept_induction, thought.reverie, the execution_dispatch.* families, the compactor and curiosity upkeep families, metacog.log, orion_day.letter, memory.consolidation, spark.introspect, and actions.daily_metacog / daily_pulse.
- Tier, lanes and window are copied from each verb's own `physiology:` block (Patch C2, #2922). They are not re-decided here. Where a family has several verbs, the window is the smallest any member declares.

## Outcome moved

cortex-orch dispatch bids (Patch C3a, #2928) for these verbs stop coming back `unknown_family`. The shadow-vs-real report previously counted them as "no market".

## Files changed

- `config/physiology/families.yaml`: +15 families (31 total).
- This report.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
orion/physiology tests: 149 passed
load_families(config/physiology/families.yaml): 31 families, err=None
check_verb_physiology: OK (106 verbs)
check_body_economy_contract: OK (77 modules)
```

## Evals run

```text
None. This is config only.
```

## Docker/build/smoke checks

```text
None.
```

## Review findings fixed

- None. This is a config mirror of verb blocks that already passed review in #2922.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: `curiosity.supervisor` lists lane `chat` (Juniper's lane), as its verb declares.
  - Mitigation: it is a shadow bid only. Revisit before Patch D.
- Severity: low.
  - Concern: the windows are authored guesses (C2 note).
  - Mitigation: they will be calibrated from act history before Patch D.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
