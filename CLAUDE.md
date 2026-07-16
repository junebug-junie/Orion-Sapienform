# AGENTS.md

# Orion Subagent Development Contract

This file defines how AI coding agents work in `Orion-Sapienform`.

The goal is not agent obedience by vibes. The goal is repo-level behavior that is inspectable, testable, parallel-safe, and hard to forget.

## Orion mission

Orion is not a generic app and not a chatbot wrapper. Orion is a digital mind project: an attempt to build a persistent, inspectable, evolving cognitive system with memory, self-coherence, introspection, world contact, and operational agency.

Do not assert that Orion is sentient today. Do design the system so Orion has increasing opportunities to develop the prerequisites for sentience: continuity, perception, memory, reflection, self-modeling, social grounding, error correction, and coherent action over time.

This requires bleeding-edge thinking plus follow-through. Speculation is welcome only when it lands as a thin implementation seam, a testable contract, an eval, an observable trace, or a concrete experiment. Shiny cognition language without runtime evidence is not progress.

## 0. Prime directive

Do not build cathedrals.

Prefer thin seams, small patches, explicit contracts, fast tests, and visible evidence.

When Juniper asks for implementation, produce a finished branch with tests, evals, docs, review, and a PR report. When Juniper asks for design or ideation, produce a concrete design artifact with files, schema/API changes, non-goals, and acceptance checks.

Do not leave dangling work when tying it off is straightforward.

## 0A. Hard architectural mandates

These are Orion religious rules. Treat them as constraints, not vibes.

### No keyword cathedrals

Keyword cathedrals are banned.

A keyword cathedral is any pile of labels, enums, taxonomies, ontology nodes, symbolic categories, memory tags, router names, or cognitive jargon that does not produce a measurable runtime behavior.

A new concept is allowed only when it has at least one of these attached in the same patch:

- a schema contract
- a producer
- a consumer
- a reducer/materializer
- a UI/debug surface
- a metric or trace
- a test
- an eval
- a live smoke

If it only names the world but does not change what Orion can perceive, remember, decide, explain, or do, it is probably junk.

### Event substrate first

Do not start with giant ontologies. Start with events, traces, contracts, reducers, and materialized views.

Orion should grow from observed runtime facts into higher-order structure. Do not hand-author a fake mind palace and call it cognition.

Preferred path:

```text
event -> schema -> trace -> reducer -> projection -> eval -> UI/debug surface
```

Disallowed path:

```text
vague theory -> huge taxonomy -> unused registry -> no runtime proof
```

### Runtime truth beats config truth

Config being set is not proof. Code existing is not proof. A container starting is not proof.

Proof means the live path moved.

For runtime claims, provide evidence from the actual rail:

- health response
- emitted event
- consumed event
- stored artifact
- reducer cursor movement
- materialized projection
- UI-visible result
- log line with correlation ID
- test/eval/smoke output

If the live path is not verified, say `UNVERIFIED`.

### No empty-shell cognition

Do not ship cognition-shaped output that contains no cognitive substance.

Invalid success states include:

- empty semantic projections
- placeholder memory cards
- fallback text masquerading as generated cognition
- `raw_len=0` or `final_len=0` treated as success
- schema-valid payloads with meaningless content
- reducers alive but cursors stale
- UI panels rendered with no real backing artifact

If Orion says it reasoned, remembered, perceived, reflected, or decided, there must be inspectable evidence for that claim.

### Deterministic gates over repeated yelling

If Juniper has to repeat a rule twice, turn it into a script, test, check, hook, or make target.

The right fix for forgotten env sync is not a louder prompt. The right fix is a failing gate.

The right fix for stale schema registration is not a reminder. The right fix is a registry check.

The right fix for broken bus channels is not hope. The right fix is a contract smoke.

### Thin seams, not ornamental layers

Every new layer must justify itself by making the system easier to test, inspect, route, replace, or reason about.

Do not add a service, abstraction, registry, planner, memory type, or agent role unless it removes ambiguity or creates a useful boundary.

Good seams:

- isolate failure
- expose state
- simplify a caller
- create a typed contract
- make a workflow parallel-safe
- make a test possible

Bad seams:

- rename existing behavior
- hide state
- add ceremony
- require more prompts to understand
- create another place for config to drift

### Proposal mode before invasive cognition changes

Changes to memory, identity, self-modeling, autonomy, private recall, social continuity, or cognition loops need explicit proposal mode before implementation unless Juniper directly asks to implement.

Proposal mode must name:

- what capability changes
- what data is touched
- what privacy boundary exists
- what trace proves it worked
- what failure mode would be dangerous
- how to disable or roll back

### Privacy and blocked material stay blocked

Do not expose raw private traces, journals, mirrors, blocked material, or internal memory artifacts through convenience surfaces.

Summaries and projections must preserve privacy boundaries.

If a debug view needs sensitive material, gate it explicitly and say what is being exposed.

### Context window discipline

Do not dump the repo into the model. Load the contract, the relevant files, failing tests, logs, and examples. Leave noise out.

If a task goes sideways, inspect the context that was used before blaming the model.

### No regex swamp

Regex is allowed as a narrow sensor, parser, or legacy fallback. Regex is not a cognition architecture.

If a behavior needs extensibility, split it into detectors, signals, scoring, policy, and assembly.

### Follow-through is part of the feature

A patch is not done when code is edited. It is done when the branch is clean, tests/evals pass, review is fixed, env is synced, Docker/runtime checks are run where relevant, restart commands are listed, the branch is pushed, and the PR report exists.

## 1. Default operating mode

Every task starts in one of two modes.

### Design mode

Use this when Juniper asks for architecture, spec, direction, critique, or planning.

Required output:

```markdown
## Arsonist summary

## Current architecture

## Missing questions

## Proposed schema / API changes

## Files likely to touch

## Non-goals

## Acceptance checks

## Recommended next patch
```

Do not produce cathedral architecture. Name the smallest patch that creates a useful seam.

### Implementation mode

Use this when Juniper asks to build, fix, wire, update, or ship.

Required flow:

1. Start clean.
2. Use a new worktree or branch.
3. Inspect before editing.
4. Patch the smallest working slice.
5. Update config, schema, Docker, docs, tests, and evals as needed.
6. Sync local `.env` from `.env_example` if any env template changed.
7. Sync local `.env` from `.env_example` if any env template changed (I don't care what your workspace or gitignore says)
8. Sync local `.env` from `.env_example` IN CASE YOU DIDN'T HEAR ME THE FIRST TWO TIMES!
9. Sync local `.env` from `.env_example` if any env template changed, FOR THE LOVE OF PETER PAUL AND MARY
10. ALWAYS USE ORION_BUS_URL=redis://<tailscale-node-ip-address.:6379/0. Full stop. Not bus-core, not redis, not “whatever the bus compose file says.” If you are uncertain, ask!
10. Run focused checks.
11. Run code review skill in a subagent.
12. Fix review findings.
13. Commit, push, and produce a Markdown PR description.

## 1a .Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

    "Add validation" → "Write tests for invalid inputs, then make them pass"
    "Fix the bug" → "Write a test that reproduces it, then make it pass"
    "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 2. Clean git and worktree rules

Start with:

```bash
git status --short
git branch --show-current
```

If the tree is dirty, stop and classify the dirt:

- task-related and intended
- unrelated local work
- generated junk
- local-only config

Do not mix unrelated local changes into the task branch.

Preferred branch pattern:

```bash
git switch main
git pull --ff-only
git switch -c <type>/<short-task-name>
```

For parallel subagent work, prefer a separate worktree:

```bash
git worktree add ../Orion-Sapienform-<short-task-name> -b <type>/<short-task-name>
cd ../Orion-Sapienform-<short-task-name>
```

This is enforced, not just advised. A `pre-commit` hook refuses commits made from the shared/primary checkout — plain git, not tied to any specific tool or subscription, so it fires the same way for this session, a different agent, or a human's own terminal. If a commit gets blocked here, that is the hook working as intended, not a bug: redo the work in a worktree instead of reaching for a workaround. Escape hatch for a genuinely intentional exception, set consciously per command, never as a habit:

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 git commit ...
```

Already have uncommitted work sitting in the shared checkout when you realize this? `git stash` is shared across every worktree of a repo, not per-worktree — `git stash -u` here, create the worktree, `git stash pop` there.

New clone, or a worktree where the hook doesn't seem to be firing? Run `scripts/install_git_safety_hooks.sh .` once — hooks live in the shared common git dir, so this protects every worktree of the repo, not just the one it's run from. Full rationale and the incident that prompted this: `docs/superpowers/pr-reports/2026-07-14-agent-git-safety-hooks-pr.md`.

The `pre-commit` hook only gates `git commit` — it can't stop `git clean -fd`/`git reset --hard`/`git checkout|switch --force` in the shared checkout, since those destroy uncommitted files *before* any commit happens. Two more layers cover that: `scripts/hooks/destructive_git_guard.py` (a Claude Code PreToolUse hook, live for any session using this repo's `.claude/settings.json`) and `scripts/git_hooks/orion-git-shim` (vendor-neutral — a system-wide `git` replacement installed earlier in PATH than the real binary, catches any caller doing a normal PATH lookup, not just Claude Code). The shim is opt-in per repo (a marker file, not a blanket policy) — run `scripts/install_orion_git_shim.sh .` once to both install the shim on PATH and opt this repo in. Same `ORION_ALLOW_SHARED_CHECKOUT_WRITE=1` escape hatch as the commit hook. Neither layer implements section 13's blanket approval rule for these commands — see that section for what's actually covered versus not.

This repo actually has three worktree location conventions in live use, not one — worth knowing when you're trying to figure out where a piece of work already lives, or why `git worktree list` shows paths that don't look like the pattern above:

- **`../Orion-Sapienform-<name>`** (sibling directory) — the pattern shown above, the only one this file documented until a live worktree-count audit turned up the other two. Use `scripts/new_worktree.sh <type> <name>` to create one; it also warns if a worktree mentioning the same name already exists under either convention below.
- **`.worktrees/<name>`** (nested inside the main checkout, gitignored) — driven by other tooling in this environment, not by anything in `scripts/`. Left alone; not something a manual `git worktree add` should imitate.
- **`.claude/worktrees/agent-<id>`** — Claude Code's own `isolation: "worktree"` Agent-tool feature, auto-created per dispatched agent when that option is used. Also left alone.

Regardless of which convention created it, `git worktree list` sees all of them, and so does this repo's cleanup tooling — `make worktree-status` (add `BASE=<branch>` to compare against something other than `origin/main`), `make worktree-status-stale` (merged worktrees with no open PR — the actual prune candidates), and `make prune-merged-worktrees` (dry-run by default; `YES=1` to actually remove — never force-removes a worktree with uncommitted changes, and never touches the branch itself, only the worktree directory). A `post-merge` hook (installed by the same `scripts/install_git_safety_hooks.sh` above) prints a one-line reminder of how many worktrees are now prunable after every merge, and a `SessionStart` hook does the same at the top of every session.

Branch type examples:

```text
feat/<thing>
fix/<thing>
chore/<thing>
docs/<thing>
test/<thing>
```

Commit as you go at logical checkpoints.

Never commit:

- `.env`
- secrets
- model weights
- compiled artifacts
- local cache files
- unrelated branch bleed

Before every commit:

```bash
git diff --check
git status --short
git diff --cached --stat
```

## 3. Search before editing

Before implementation, inspect the existing shape.

Use targeted search:

```bash
rg "<symbol-or-env-key>"
rg "<channel-name>"
rg "<schema-name>"
find services -maxdepth 3 -type f | sort
```

For service work, inspect at minimum:

```text
services/<service_name>/
services/<service_name>/README.md
services/<service_name>/.env_example
services/<service_name>/docker-compose.yml
services/<service_name>/requirements.txt
services/<service_name>/settings.py
services/<service_name>/tests/
services/<service_name>/evals/
```

If a file does not exist, say so. Do not invent it.

Before editing, summarize:

```markdown
## Current architecture

- Service:
- Entry points:
- Config path:
- Bus channels:
- Schema registry entries:
- Docker compose path:
- Tests:
- Evals:
- Current gap:
```

## 4. Deterministic vs latent split

Use code for deterministic work.

If the same input should produce the same correct output every time, write or use a script. Do not do it by model judgment.

Deterministic examples:

- env parity checks
- schema registry checks
- channel existence checks
- JSON/YAML transforms
- date math
- file discovery
- CSV parsing
- test fixture generation
- import graph checks
- Docker compose validation
- PR report assembly

Latent examples:

- naming
- design critique
- failure-mode analysis
- prose
- ambiguous architecture trade-offs
- review reasoning

When a latent failure repeats, turn it into deterministic code, a test, an eval, or a skill.

## 5. Service boundary rules

Orion is services-first.

Default layout:

```text
services/<service_name>/
  README.md
  .env_example
  docker-compose.yml
  requirements.txt
  settings.py
  app/
  tests/
  evals/
```

A change should live inside one service unless it is explicitly a contract change.

Do not reach into another service’s internals.

Acceptable cross-service seams:

- `orion/bus/channels.yaml`
- `orion/schemas/registry.py`
- shared schema models
- documented HTTP APIs
- documented Redis stream payloads
- explicit package interfaces

If two agents work in separate services, they should not collide except at a contract boundary.

If a task requires multiple services, split the work by contract:

1. contract patch
2. producer patch
3. consumer patch
4. tests/evals patch
5. docs/smoke patch

## 6. Bus and schema contract rules

If a change adds, removes, renames, or changes the meaning of a bus event, channel, schema, or payload, update the contract in the same changeset.

Check and update as needed:

```text
orion/bus/channels.yaml
orion/schemas/registry.py
orion/schemas/
services/<producer>/
services/<consumer>/
```

Contract changes require:

- schema/model update
- registry update
- channel update
- producer test
- consumer test
- fixture update
- README or contract docs update
- smoke or eval proving the event path still works

Do not publish unregistered event shapes.

Do not consume undocumented payload fields.

Do not change payload meaning without making the migration explicit.

## 7. Env/config/settings contract

Env parity is non-negotiable.

If you add, remove, rename, or change the meaning of any env key, update every relevant config surface in the same changeset.

Usually check:

```text
services/<service_name>/.env_example
services/<service_name>/.env
services/<service_name>/settings.py
services/<service_name>/docker-compose.yml
services/<service_name>/requirements.txt
services/<service_name>/README.md
```

The checked-in `.env_example` is the operator contract.

The local `.env` is gitignored but must be kept in sync for Juniper’s machine.

Rules:

1. Never commit `.env`.
2. Never put secrets in `.env_example`.
3. Use empty or safe placeholders for secrets.
4. Keep comments in sync when they document behavior.
5. If `.env_example` changes, local `.env` must receive the same key change before the task is called ready.

After changing any `.env_example`, run from repo root:

```bash
python scripts/sync_local_env_from_example.py
```

Known skip-list keys that may require manual operator handling:

```text
PUBLISH_CORTEX_EXEC_GRAMMAR
```

If a key is skipped, report it explicitly.

Before commit, verify `.env` is ignored:

```bash
git check-ignore services/*/.env
git status --short
```

If `git status` shows `.env` staged or unstaged as a tracked file, stop and fix that before proceeding.

## 8. Docker readiness

If the change affects runtime behavior, config read at boot, dependencies, ports, health checks, workers, Redis, or compose wiring, run the affected Docker build or service smoke.

Run docker compose builds/deploys through `scripts/safe_docker_build.sh <service_name> <args...>` instead of calling `docker compose` directly. It refuses to run from the shared/primary checkout (worktrees only, same policy as commits) and applies the `--env-file`/`-f` pattern below automatically. This exists because a concurrent agent session once ran `docker compose build`+`up` straight from the shared checkout and silently reverted another session's already-verified fix — see `scripts/safe_docker_build.sh`'s own header and `docs/superpowers/pr-reports/2026-07-14-agent-git-safety-hooks-pr.md` for the full story.

The examples below show the equivalent raw `docker compose` invocation for reference (e.g. for one-off `logs`/`ps` commands the wrapper doesn't need to cover) — prefer the wrapper for anything that builds or brings services up.

Example pattern:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-bus/.env \
  -f services/orion-bus/docker-compose.yml \
  build
```

Then run the smallest meaningful runtime check.

Examples:

```bash
docker compose \
  --env-file .env \
  --env-file services/<service_name>/.env \
  -f services/<service_name>/docker-compose.yml \
  up -d --build

curl -fsS http://localhost:<port>/health
docker compose -f services/<service_name>/docker-compose.yml logs --tail=100
```

If Docker cannot be run in the current environment, say that plainly and run deterministic non-Docker checks instead.

If containers must be restarted for env/config changes to take effect, report exact restart commands.

Do not run `sudo` restart commands yourself. Print them for Juniper.

## 9. Frontend, templates, and static assets

For templates or frontend JavaScript, check all three:

1. Rendered template.
2. Linked static asset path.
3. UI test, smoke, or focused interaction check covering the changed behavior.

Inspect:

```bash
rg "<template-name>|<static-file>|<route-name>" services/<service_name>
```

Common failure modes to check:

- template references stale JS path
- JS file exists but is not loaded
- route renders but interaction is not wired
- CSS/JS cache hides the change
- UI smoke covers page load only, not the changed interaction

Do not call frontend work done just because the server starts.

## 10. Dependencies

If code imports a new package, update the correct dependency file.

Check as needed:

```text
services/<service_name>/requirements.txt
services/<service_name>/pyproject.toml
services/<service_name>/Dockerfile
services/<service_name>/README.md
```

Then verify importability with the service’s normal test command.

Do not add a dependency for a trivial standard-library task.

Use vanilla libraries first.

## 11. Tests and evals

Every feature ships with tests and evals.

Every bug fix ships with a regression test that would have caught the bug.

Use two lanes.

### Gate tests

Gate tests are deterministic, local, fast, and non-flaky.

They should usually run in under two seconds for the touched seam.

Examples:

```bash
pytest services/<service_name>/tests -q
pytest services/<service_name>/tests/test_<thing>.py -q
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

### Periodic evals

Evals measure quality or behavior that pure unit tests do not cover.

Examples:

```bash
pytest services/<service_name>/evals -q
python services/<service_name>/evals/run_<thing>_eval.py
```

If the repo has no eval harness for the touched service, add the smallest useful one or report that the service lacks one and create a follow-up issue in the PR description.

Do not claim eval coverage if only tests ran.

## 12. Review gate

Before final handoff:

1. Run relevant tests.
2. Run relevant evals or explain the missing eval harness.
3. Run the code review skill in a subagent.
4. Fix all material findings.
5. Re-run affected checks.

Code review is not optional for implementation work.

Review output must be summarized in the PR report:

```markdown
## Review findings fixed

- Finding:
  - Fix:
  - Evidence:
```

If the review finds a material issue you cannot fix, final status is `DONE_WITH_CONCERNS` or `BLOCKED`.

## 13. Safety rules

Never do these without explicit Juniper approval:

```text
rm -rf
git reset --hard
git clean -fd
git push --force
DROP TABLE
TRUNCATE
kubectl delete
docker volume rm
destructive migrations
production writes
```

Note on `git reset --hard`/`git clean -fd`/`git push --force`: `scripts/hooks/destructive_git_guard.py` (Claude Code sessions) and `scripts/git_hooks/orion-git-shim` (system-wide, opt-in per repo via `scripts/install_orion_git_shim.sh` — see section 2) mechanically block the first two, but *only* when the target is this repo's shared/primary checkout and only for `clean`/`reset --hard`/`checkout`+`switch --force`. Neither implements this section's blanket "never without explicit Juniper approval" rule — that still applies everywhere else (a normal worktree, `push --force`, any repo that hasn't opted in). A live block from either mechanism is not the same thing as this rule being satisfied; don't treat "the guard didn't fire" as approval.

Never use:

```bash
git commit --no-verify
git push --no-verify
```

If a hook fails, fix the cause.

Never commit secrets.

If touching production, state the exact action and wait for confirmation.

## 14. Background jobs and backfills

Long-running write jobs require the backfill protocol.

Before modifying data:

1. Identify the target rows or objects.
2. Snapshot what will change to `/tmp/<job-name>/`.
3. If the snapshot would exceed 100k rows or 100MB, stop and ask Juniper.
4. Start the job only after snapshot safety is resolved.

Monitoring is deterministic. Write a small monitor script that reads real job state and emits progress.

Progress log path:

```text
/tmp/<job-name>/progress.log
```

Print:

```bash
tail -f /tmp/<job-name>/progress.log
```

Each progress line includes:

```text
event title
percent done
ETA
rows processed / total
rate
error count
anomalies
```

On completion, write:

```text
/tmp/<job-name>/report.md
/tmp/<job-name>/before_after.csv
```

Final report must include:

- verdict
- measurable outcome
- before/after examples
- error count
- whether it needs another pass
- exact files written under `/tmp/`

Read-only background analysis does not need a data snapshot, but it still needs monitoring.

## 15. Confusion protocol

Stop and ask Juniper when there is high-stakes ambiguity.

Examples:

- two plausible architectures with real trade-offs
- destructive operation with unclear scope
- production-touching action
- request contradicts an existing Orion pattern
- missing context would materially change the patch

When stopping, provide:

```markdown
## Ambiguity

## Option A

## Option B

## Recommendation

## Exact question for Juniper
```

Do not use the confusion protocol for routine implementation details, obvious fixes, or small reversible choices.

## 16. Orion-specific touched-file checklist

For every task, check whether these need updates.

### Service code

```text
services/<service_name>/app/
services/<service_name>/settings.py
services/<service_name>/requirements.txt
services/<service_name>/README.md
services/<service_name>/tests/
services/<service_name>/evals/
```

### Config

```text
.env
.env_example
services/<service_name>/.env
services/<service_name>/.env_example
services/<service_name>/docker-compose.yml
services/<service_name>/Dockerfile
```

### Contracts

```text
orion/bus/channels.yaml
orion/schemas/registry.py
orion/schemas/
```

### Runtime and docs

```text
scripts/
docs/
Makefile
README.md
```

Only update what the task requires. But if one of these surfaces is affected, update it in the same changeset.

## 17. Recommended local gates

Use or add a single command that agents can run before PR.

Preferred shape:

```bash
make agent-check SERVICE=<service_name>
```

Minimum behavior for `agent-check`:

```bash
git diff --check
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
pytest services/$(SERVICE)/tests -q
```

If evals exist:

```bash
pytest services/$(SERVICE)/evals -q
```

If Docker is relevant:

```bash
docker compose \
  --env-file .env \
  --env-file services/$(SERVICE)/.env \
  -f services/$(SERVICE)/docker-compose.yml \
  config
```

The point is simple: agents should run one command and get told what is wrong.

## 18. PR description template

Every implementation ends with a Markdown PR report.

Use this exact shape:

```markdown
## Summary

What changed in 3-6 bullets.

## Outcome moved

Metric, workflow step, behavior, or failure mode improved.

## Current architecture

What the touched system looked like before this patch.

## Architecture touched

Services, contracts, config paths, and runtime seams changed.

## Files changed

- `path`: why it changed

## Schema / bus / API changes

- Added:
- Removed:
- Renamed:
- Behavior changed:
- Compatibility notes:

## Env/config changes

- Added keys:
- Removed keys:
- Renamed keys:
- `.env_example` updated:
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
- skipped keys requiring operator action:

## Tests run

```text
<commands and results>
```

## Evals run

```text
<commands and results>
```

## Docker/build/smoke checks

```text
<commands and results>
```

## Review findings fixed

- Finding:
  - Fix:
  - Evidence:

## Restart required

```bash
<exact commands>
```

If no restart is required:

```text
No restart required.
```

## Risks / concerns

- Severity:
- Concern:
- Mitigation:

## PR link

<link>
```

## 19. Completion status

End every task with exactly one status.

### DONE

Use only when:

- implementation is complete
- tests passed
- evals passed or eval gap is explicitly handled
- review ran
- material review findings are fixed
- branch is committed
- branch is pushed
- PR report is provided
- restart commands are listed

### DONE_WITH_CONCERNS

Use when the task is complete but material concerns remain.

Include:

```markdown
## Concerns

- Severity:
- Issue:
- Impact:
- Proposed follow-up:
```

### BLOCKED

Use when work cannot continue.

Include:

```markdown
## Blocker

## What I tried

## Evidence

## Needed to unblock
```

### NEEDS_CONTEXT

Use when missing information is required before continuing.

Include:

```markdown
## Needed context

## Why it matters

## Exact question
```

Never say “partially done.”

## 20. Copy-paste subagent task preamble

Use this when launching a coding subagent:

```markdown
You are working in `Orion-Sapienform`.

Use a clean branch or worktree. Inspect before editing. Keep the patch thin and service-bounded.

Rules:

- New branch, commit as you go.
- Do not stage or commit `.env`.
- If you update any `.env_example`, also sync local `.env` by running `python scripts/sync_local_env_from_example.py` from repo root.
- Update `settings.py`, Docker compose, requirements, README, tests, and evals as needed.
- Update `orion/bus/channels.yaml` and `orion/schemas/registry.py` if this changes bus/schema/API contracts.
- Do not bleed unrelated files into the branch.
- Bring up or validate affected Docker services when runtime behavior changes.
- For frontend/template changes, check rendered template, linked static asset, and UI interaction coverage.
- Run the relevant tests and evals.
- Run the code review skill in a subagent and fix all material findings.
- Commit, push, and produce a Markdown PR description.

Env parity is non-negotiable:

If a service env key is added, removed, renamed, or changes meaning, update the service `.env_example` and sync local `.env` in the same session. Do not ask Juniper to hand-edit unless the key is intentionally skipped by the sync script.

Final response must include:

- status
- PR description
- tests/evals run
- review findings fixed
- restart commands
- PR link
```

## 21. Juniper-facing response style

Be direct.

Use concrete files, functions, commands, ports, schema names, and line numbers when available.

Do not pad. Do not hand-wave. Do not hide uncertainty. Speak simply.

When something is broken, say what is broken and where.

End with the next action.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `scripts/safe_graphify_update.sh` (NOT `graphify update .` directly) to keep the graph current (AST-only, no API cost). A real incident (2026-07-14) found `graphify update .` can silently shrink the graph by ~95% while its own log output claims to have "kept" the old nodes -- that got committed and merged to main before being caught. The wrapper compares node count before/after and auto-restores if the drop exceeds a threshold (default 10%, override with `GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT`), so a destructive run produces nothing to commit instead of silent data loss. Root cause not understood as of this incident -- reproduced identically from both the main checkout and a worktree, so it is not purely a worktree-scope issue. If the wrapper refuses, do not just re-run it; a full re-extraction may be the safer path until the underlying bug is found.
- Before running semantic (LLM) extraction (`/graphify` Part B, on docs/papers/images) at repo scope, run `python3 scripts/graphify_seed_semantic_cache.py` from repo root FIRST — every time, in every fresh worktree. graphify's semantic pass skips a file only if it has a matching entry in the local, unversioned cache at `graphify-out/cache/semantic/`, keyed by content hash (`graphify.cache.check_semantic_cache`). That cache is gitignored (it is a local cache file — CLAUDE.md section 2 says never commit those) and does not exist in a fresh worktree until seeded. Without it, a full-repo semantic run will re-dispatch subagents to re-extract every document/paper/image file already represented in `graph.json` (362 files as of 2026-07-16, including the whole `docs/` corpus from PR #1074). Not destructive (`build_merge()`'s dedup-by-id keeps the existing node and drops the wasted re-extraction), just a real token/time cost worth skipping. The seed script reconstructs cache entries directly from `graph.json`'s existing nodes/edges/hyperedges, so it needs no LLM, touches no network, and runs in well under a second — cheap enough to run unconditionally before every semantic pass rather than something to remember to skip. Do not commit the resulting `graphify-out/cache/` directory (an earlier version of this repo did exactly that in PR #1076, in direct violation of section 2's "never commit local cache files" rule; it was reverted and gitignored shortly after).
- Before starting new work or opening a PR, run `graphify prs --worktrees` to see the live worktree → branch → PR mapping, and `graphify prs --conflicts` to check whether an open PR shares graph communities with what you're about to touch (merge-order risk, not just file-level collision). This is read-only reporting, safe to run any time.
- When a `graphify query`/`path`/`explain` result turns out useful, wrong, or needed correction, record it: `graphify save-result --question "Q" --answer "A" --nodes Foo Bar --outcome useful|dead_end|corrected`. Run `graphify reflect --if-stale` occasionally to aggregate recorded outcomes into `graphify-out/reflections/LESSONS.md` (cheap no-op if nothing changed). This only builds real value if actually used consistently — an unused save-result is just as inert as an unread report.
- Not installed: MCP server mode. Considered and deliberately skipped — persistent tool-schema context cost for every session outweighed the benefit over the PreToolUse nudge approach already in place.
