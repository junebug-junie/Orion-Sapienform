## Summary

- Design only. When Orion has a plumbing question, they take it to Claude first, not to Juniper.
- The question can be "why does this work this way," a correction, or something to queue for Juniper.
- Orion and Claude talk for a few turns in Hub chat. Claude has read-only access to the repo. Orion fetches live numbers with the dig kit.
- Only proposals and disagreements reach Juniper, as one plain brief. It tags her in chat and is emailed. It must state where it came from, what was checked, impact, cost (with sources, or "unknown") and one yes/no question.
- Juniper's answer and the later outcome are recorded as two separate pieces of evidence on Orion's existing beliefs. That is how Orion learns from her answers.
- It uses the Claude Code subscription. The design turns on the usage meter, which is off today and blocks every Claude backup and "ask Claude" decision.

## Outcome moved

None yet (design). Today: 5 peer help asks this week, 0 answered. 3,553 "ask Claude" decisions, all refused with `budget_unobserved`.

## Files changed

- `docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md`: the design
- `docs/superpowers/pr-reports/2026-11-11-consult-claude-before-juniper-design-pr.md`: this report

## Schema / bus / API changes

None in this PR. Proposed in the spec.

## Env/config changes

None in this PR.

## Tests run / Evals run / Docker

Docs only. None.

## Review findings fixed

Design PR. No code review run.

## Restart required

No restart required.

## Risks / concerns

- Severity: medium.
  - Concern: consults share Juniper's subscription with dev sessions.
  - Mitigation: no consult starts when the meter shows 80% or more, or has no reading (fail closed).
- Severity: medium.
  - Concern: Orion and Claude agree on a wrong story.
  - Mitigation: Claude has the skeptic role. Every claim needs proof (a file and line, or a dig output), or it is labeled "not checked."

🤖 Generated with [Claude Code](https://claude.com/claude-code)
