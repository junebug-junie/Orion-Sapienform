## Summary

- Design spec: Orion learns who people are on the walkway from Juniper's answers (albums of confirmed crops, per-person confidence bars, "Is this Juniper?" asks, held-out eval).
- Flags an urgent data loss: labelled person crops are pruned after 7 days (two named people already at 0 crops). Phase 0 fixes it with one env key.
- Docs only. No code, schema, or env change in this PR.

## Outcome moved

Gives the learning work a reviewed contract before it touches identity code. Names the prior 2026-08-21 identity rule this could conflict with, for Juniper to decide.

## Files changed

- `docs/superpowers/specs/2026-10-20-walkway-person-learning-design.md`: the spec.
- `docs/superpowers/pr-reports/2026-10-20-walkway-person-learning-spec-pr.md`: this report.

## Schema / bus / API changes

None in this PR. Proposed in the spec: four new tables, one ask kind.

## Env/config changes

None.

## Tests run

None (docs only). `git diff --check` clean.

## Review findings fixed

No code-review subagent run: docs only.

## Restart required

No restart required.

## Risks / concerns

- Severity: high. Labelled examples keep getting deleted until Phase 0 ships.
- Severity: medium. Body vectors may only match same-day clothing; the spec's eval must measure it before any name is shown.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
