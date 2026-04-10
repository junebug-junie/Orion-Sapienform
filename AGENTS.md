# Orion agent contract

## Purpose
This repository contains multiple services under `services/`.
Your job is to make the smallest correct change, verify it against the real affected runtime path, and report evidence without bloat.

## Non-negotiable rule
Do not claim a fix unless you ran verification against the runtime code path you changed.

If verification did not run successfully, the final status must be exactly:

UNVERIFIED

## Scope discipline
- Infer the target service from the files touched.
- If files are under `services/<service_name>/`, treat that as the primary service in scope.
- Stay inside that service unless the change clearly requires cross-service edits.
- Do not perform broad repo refactors unless explicitly requested.
- Do not “improve” unrelated files.

## Service rule for anything in root/services
When working on any service under `services/`:
1. Read the entry path and the immediate runtime path first.
2. Prefer the smallest patch that fixes the problem.
3. Verify using the smallest command that exercises the changed path in that service.
4. Report exact evidence, not confidence language.

## Runtime-path-first rule
Before summarizing, inspect the actual files on the execution path you changed.

Examples of execution-path files include:
- service entrypoints
- route handlers
- orchestrators
- supervisors
- workflow runners
- templates actually rendered by the service
- JS files actually loaded by the page
- test files covering the changed rail
- docker-compose or env wiring if startup/runtime behavior depends on them

Do not infer behavior from adjacent helper files alone.

## Smallest-proof rule
Use the narrowest verification that proves the change.

Preferred order:
1. targeted unit/integration test for the changed behavior
2. smallest service-local runtime command
3. compile/typecheck/lint for changed files
4. broader suite only if required

Avoid repo-wide test sweeps unless the task truly spans multiple services.

## Change budget
- Make the smallest viable patch.
- Preserve existing style and architecture unless the task requires otherwise.
- No speculative abstractions.
- No placeholder code unless explicitly requested.
- No TODOs as a substitute for implementation.

## Truthfulness rules
- Never say “should be fixed”.
- Never say “likely fixed”.
- Never imply runtime success from static code reading.
- If a command fails, say so plainly.
- If the environment prevents verification, print `UNVERIFIED` and name the blocker in one sentence.

## Required adjacency checks
When you change one of the following, also inspect the obvious adjacent wiring:

### Python backend logic
Check:
- service entrypoint
- imports
- relevant tests

### Templates or frontend JS
Check:
- rendered template
- linked static asset
- any UI test covering the changed interaction

### Env/config/settings
Check whether the change also requires updating:
- `.env_example`
- service `.env`
- `docker-compose.yml`
- settings/config loaders

### Bus/schema/channel changes
Check whether the change also requires updating:
- schema registry references
- channel constants
- producers/consumers
- tests asserting the contract

## Command policy
When verifying, prefer commands scoped to the touched service.

Examples:
- `python -m compileall services/<service_name>`
- `pytest -q services/<service_name>/tests/<target>`
- `npm test -- <target>`
- a targeted service-local run command or integration command

Do not run destructive commands unless explicitly requested.

## Required final response format
### Summary
One short paragraph.

### Files changed
Exact file paths only.

### Verification
For each command:
- exact command
- exit code
- concise observed result

### Remaining risks
Only real remaining uncertainty.

## Forbidden response patterns
- long design essays when implementation was requested
- repeating the prompt back to the user
- claiming verification without commands and results
- listing hypothetical issues not grounded in the changed code
- padding with generic best practices

## Closure requires
For bugs reported from live logs, mocked or service-local harnesses do not count as closure unless the user explicitly asks for a harness only.
Closure requires either:
1. a live-stack verification script with pass/fail evidence, or
2. a direct runtime reproduction with exact commands and exact evidence lines.
Otherwise reply UNVERIFIED.

## Preferred style
Be terse, concrete, and evidence-first.
