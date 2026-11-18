## Summary

- The privacy gate (`tests/test_no_personal_terms.py`) failed on main since commit 1f8f7ad4f: the Situation Graph design spec named Juniper's home city.
- Replaced the city with "her home city" / "the home-city case". The finding the spec describes (a confabulated "correction" row) is unchanged.

## Outcome moved

`orion-static-gates` goes green again on main and on every open PR (it was red on 3 main commits in a row, blocking PR #3002's checks).

## Files changed

- `docs/superpowers/specs/2026-10-07-situation-graph-design.md`: removed the home-city name (2 lines).

## Tests run

```text
pytest -q tests/test_no_personal_terms.py   7 passed
```

## Evals run

None; docs-only change.

## Review findings fixed

None; 2-line docs scrub, no code. Review skipped as not material.

## Restart required

No restart required.

## Risks / concerns

- Severity: low. Concern: the city name stays in git history. Mitigation: out of scope; the gate guards the tree, not history.
