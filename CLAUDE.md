# Orion Repo Instructions

You are working in Orion-Sapienform.

## Operating posture

Do not build cathedrals. Prefer thin, inspectable, testable seams.

For Knowledge Forge:
- Repo/files remain authority.
- Accepted claims/specs/decisions must not be silently mutated.
- Generated ideas go to reviews/pending or context_packs/generated.
- Disputed/stale claims are excluded from execution context by default.
- Ideation output must be source-grounded and marked as proposal, not truth.

## Workflow

Before implementation:
1. Inspect existing files with rg/find.
2. Summarize current architecture.
3. Propose a minimal patch plan.
4. Wait before destructive changes unless explicitly told to implement.

## Preferred outputs

For design/ideation tasks, produce:
- arsonist summary
- missing questions
- proposed schema/API changes
- files likely to touch
- non-goals
- acceptance checks
