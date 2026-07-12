## Summary

- Phase 4 of the inner-state unification plan: a read-only research cross-check between two independently-computed node-attribution mechanisms — `orion-attention-runtime`'s salience scoring and `orion-field-digester`'s diffusion provenance.
- Answered from 295 real ticks (~10 minutes) queried directly from live Postgres, not assumption.
- No code changed — pure research, per the plan's own Phase 4 scoping.

## Outcome moved

A concrete, data-backed answer to "should these two attribution mechanisms be unified" — they should not, and the reason why is now documented with real numbers rather than left as an open question.

## Current architecture

`orion-attention-runtime` scores per-node/capability salience (`weighted_pressure`/`urgency_score`/`confidence_from_vector`/`compute_salience`, bounded by top-5/threshold gates). `orion-field-digester` separately tracks, per capability channel, which edge source contributed the largest weighted amount this tick (`capability_provenance`). Both existed before today; neither had been checked against the other.

## Architecture touched

None — documentation only (`docs/notes/`, plan doc, one service README).

## Files changed

- `docs/notes/2026-07-12-phase4-attention-provenance-crosscheck.md` (new): the full analysis — method (real SQL against `substrate_self_state`/`substrate_field_state`), findings table, interpretation, verdict.
- `docs/superpowers/plans/2026-07-12-inner-state-unification-plan.md`: Phase 4 marked done with the verdict summarized inline.
- `services/orion-field-digester/README.md`: new section documenting `capability_provenance` and pointing at this cross-check.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

None applicable — read-only research, no code path.

## Evals run

None applicable.

## Docker/build/smoke checks

None applicable — nothing deployed, nothing to build.

## Review findings fixed

None — no code changed, so the code-review skill wasn't run. The SQL queries backing every number in the research doc were run directly against live Postgres and are reproducible (verbatim in the doc).

## Restart required

```text
No restart required.
```

## Risks / concerns

None.

## PR link

Branch pushed: `docs/attention-provenance-crosscheck-phase4`
