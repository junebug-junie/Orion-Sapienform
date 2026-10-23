# chore: CODEOWNERS for the body-economy paths

## Summary

- Adds `.github/CODEOWNERS`. It routes changes to the paths that decide what Orion may spend (heat, GPU, money, Juniper's attention) to Juniper for review.
- The paths are taken from `config/physiology/contract.yaml` `protected_paths` (PR #2862), plus `config/gpu_pool.yaml`.
- Juniper answered "sure" to the design's missing question 7 (2026-10-23).

## Outcome moved

Any PR that touches these paths automatically requests Juniper's review. Branch protection `require_code_owner_reviews` was switched on alongside this file.

Honest limit: Juniper opens these PRs and merges them herself as admin (`enforce_admins=false`), and GitHub never counts an author's own approval. So the setting routes and labels; it does not block her. The hard gate is the CI contract check in #2862.

## Files changed

- `.github/CODEOWNERS`: new.
- `docs/superpowers/pr-reports/2026-10-23-body-economy-codeowners-pr.md`: this report.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
Not applicable. GitHub validates CODEOWNERS syntax. Check: gh api repos/junebug-junie/Orion-Sapienform/codeowners/errors
```

## Evals run

```text
None.
```

## Docker/build/smoke checks

```text
None.
```

## Review findings fixed

- None. This is a config-only file. The path list was checked against `contract.yaml`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low.
  - Concern: the list can drift from `contract.yaml`.
  - Mitigation: Phase 2's full gate can assert they match (follow-up).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
