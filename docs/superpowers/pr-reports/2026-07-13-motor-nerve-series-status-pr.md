# docs: motor-nerve series status (2026-07-13)

## Summary

- Added a self-contained status doc (`docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-series-status.md`) recording where the P0–P7 endogenous-action motor-nerve series stands: P0, P1, and P6 merged; P2–P5 unstarted and correctly blocked on P1's own stated burn-in requirement; P7 gated on two weeks of clean P1–P3 burn-in regardless of today's GO verdict.
- Flagged a real gap found while writing this: the original full P0–P7 spec doc (`docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md`) never merged to main — it exists only on the still-open `docs/endogenous-action-spec` branch. This status doc is written to stand alone rather than depend on that branch landing first.

## Outcome moved

None — this is a documentation-only status record, no code, no config, no runtime change. Written so anyone landing on `main` can see the series' current state without having to reconstruct it from three separate PR reports and a not-yet-merged spec branch.

## Current architecture

No architecture touched. `EXECUTION_DISPATCH_MODE` remains `dry_run` by default; nothing about the live environment changes as a result of this doc or the three patches it summarizes.

## Files changed

- `docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-series-status.md` (new) — the status record.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

Not applicable — documentation-only change, no code touched.

## Evals run

Not applicable.

## Docker/build/smoke checks

Not applicable.

## Review findings fixed

Not applicable — no code-review pass run for a docs-only change; nothing here has a runtime surface to review.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: `docs/endogenous-action-spec` (the original full P0–P7 spec) is still unmerged. This doc references it but does not fix that gap.
- Mitigation: none attempted here by design — merging or rewriting that branch is a separate decision outside this doc's scope. Flagged explicitly so it isn't silently forgotten.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/docs/motor-nerve-series-status
