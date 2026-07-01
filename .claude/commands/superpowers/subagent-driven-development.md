You are orchestrating a parallel subagent-driven implementation sprint for Orion Sapienform.

Session focus / task scope:
$ARGUMENTS

## Before spawning

1. Read CONTINUATION.md (or equivalent spec) and the files it points to.
2. Confirm the branch is clean and main is up to date.
3. Identify which items are truly independent (can run in parallel) vs. which have ordering dependencies.
4. For each item: name the exact files to create or edit, the acceptance checks, and the minimal scope.

## Spawn strategy

- Spawn one Agent per independent implementation item. Items with ordering dependencies must be serialized.
- Each agent prompt must be self-contained: paste in the relevant spec text, the exact files to touch, the acceptance checks, and the pattern to follow (e.g. "mirror self_state_ctx.py").
- Agents write code and tests only. No docs, no migrations, no scaffolding beyond what the spec requires.
- Each agent must run the affected tests before returning and report pass/fail.

## Orchestrator responsibilities

After all agents return:
1. Read every file each agent touched. Verify the diff matches the spec — agent intent ≠ agent output.
2. Run the full relevant test suite once.
3. If any agent's output is wrong or incomplete, fix it directly rather than re-spawning.
4. Commit all changes in one focused commit per logical rung (not one giant commit).
5. Report: what shipped, what tests passed, what remains.

## Hard constraints

- No agent may touch files outside its assigned scope.
- No auto-apply of proposals, no flag flips on dangerous gates (endogenous agency, etc.).
- Every new adapter must degrade gracefully to None on absent input — never raise.
- Cap all collections (evidence_event_ids ≤ 200, receipts per episode, etc.).
- Disputed/stale nodes excluded from execution context by default.
- No invented abstractions. Ride existing seams.
