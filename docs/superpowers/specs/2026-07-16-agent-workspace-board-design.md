# Design: Vendor-neutral agent workspace board

**Date:** 2026-07-16  
**Status:** Proposed — awaiting implementation plan after Juniper reviews this spec  
**Problem class:** Agent ops continuity (not Orion cognition). Parallel to the git safety / worktree hygiene stack: turn a repeated "don't lose the fork" instruction into a script + hooks.

## Arsonist summary

Multiple agents fly concurrently on the same host. Work forks into sub-findings, review deferrals, and open decisions; those land in PR-report prose ("not fixed", "deferred") or die with the session. Agents cannot see what other agents are doing until they collide on the same service or files — which has been happening repeatedly.

This design adds a **host-local, real-time workspace board**: open items (decisions/findings/blockers), presence (thread summary + current task per worktree), and **disclosure-only** collision warnings. Vendor-neutral CLI is the core; Claude/Cursor SessionStart/Stop hooks are thin adapters. Same pattern as `orion-git-shim` + disclosure SessionStart for worktrees.

## Current architecture (ground truth)

**Built and wired**

- Vendor-neutral git safety: `scripts/git_hooks/orion-git-shim`, `pre-commit`, `scripts/hooks/destructive_git_guard.py`
- SessionStart worktree summary: `scripts/hooks/session_start_worktree_summary.py` (Claude Code only; no `.cursor/hooks.json` yet)
- Worktree tooling: `scripts/worktree_lib.py`, `make worktree-status*`, `scripts/prune_merged_worktrees.py`, `graphify prs --worktrees` / `--conflicts`
- AGENTS.md completion statuses and PR-report Concerns sections

**Does not solve this**

- `reviews/pending/` — nearly empty; not an ops ledger
- Hub Pending Decisions / Proposal Ledger — Orion cognition proposals, not agent-session state
- PR-report "not fixed" bullets — written, then orphaned; no queryable store
- No host-shared presence board; no check-in/out ritual; no Stop-hook parking of open forks

**Live signal:** many worktrees with no open PR; concurrent agents regularly collide on the same communities/services.

## Locked decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Live store location | Host-local `~/.orion/agent-board.jsonl` (create `~/.orion/` on first write) |
| Git-committed ledger | **Out of v1** (optional park/export later) |
| Multi-agent visibility | Real-time via shared host file + `fcntl` locking |
| SessionStart view | This-worktree items **+** global strip (blockers + juniper escalations) **+** other agents' presence |
| Presence write path | Hybrid **C**: hooks upsert stub from cwd; agent fills summary/task; stale after **N=30 minutes** |
| Presence close | Auto-close when worktree disappears from `git worktree list` (reconcile on checkin/SessionStart **and** prune script) |
| Collision behavior | **A** — disclosure only; no soft/hard gate in v1 |
| Vendor surface | CLI works from any terminal; Claude + Cursor adapters call the same CLI |

## Core question answered

How do agents (and Juniper) keep a durable check-in/out board for open decisions and findings, see what others are working on, and get warned about likely collisions — without depending on one AI vendor or on git commits for live truth?

## Proposed shape

### Storage

- **Path:** `~/.orion/agent-board.jsonl`
- **Format:** append-only JSONL event log; readers materialize current state by replaying (or a tiny sidecar state file rewritten under the same lock — implementation detail, must stay crash-safe).
- **Concurrency:** exclusive `fcntl` lock on a companion `.lock` file for every read-modify-write and every append that must be ordered with reconcile.
- **Not gitignored in the repo sense** — the file lives outside the repo. Repo only ships the CLI, hooks, tests, and AGENTS.md pointer.

### Record types (fixed small enums — no taxonomy cathedral)

**`presence`** (one logical row per worktree path)

| Field | Notes |
|---|---|
| `worktree_path` | Absolute path; primary key for presence |
| `branch` | From `git` at upsert |
| `pr` | Optional URL/number if known |
| `thread_summary` | ≤1 paragraph; may be empty on stub |
| `current_task` | Specific task-level string; may be empty on stub |
| `heartbeat_at` | ISO-8601 UTC |
| `status` | `active` \| `stale` \| `closed` |
| `session_id` | Opaque string from caller/hook when available |

**`item`** (open work)

| Field | Notes |
|---|---|
| `id` | Stable id (uuid) |
| `kind` | `decision` \| `finding` \| `blocker` \| `followup` \| `theme` |
| `severity` | `blocker` \| `should` \| `note` |
| `owner_scope` | `this-worktree` \| `other-worktree` \| `juniper` \| `unassigned` |
| `scope_note` | **Required** when `owner_scope != this-worktree` |
| `worktree_path` | Default: cwd worktree |
| `pr` | Optional |
| `related_files` | Optional list of paths (aids collision) |
| `parent_id` | Optional; fork trees |
| `status` | `open` \| `parked` \| `resolved` \| `handed-off` |
| `summary` | One-line or short body |
| `created_at` / `updated_at` | ISO-8601 UTC |

### CLI — `scripts/agent_board.py`

Vendor-neutral core. All hooks call this; humans and any agent tool can call it from a plain terminal.

| Command | Behavior |
|---|---|
| `checkin` | Resolve cwd → worktree; upsert presence stub; reconcile closed worktrees; print **three layers** (below); print collision warnings for cwd |
| `heartbeat [--summary TEXT] [--task TEXT]` | Refresh `heartbeat_at`; set summary/task when provided |
| `add --kind ... --severity ... --summary ... [--scope ...] [--scope-note ...] [--parent ...] [--files ...]` | Append open item |
| `resolve ID [--status resolved\|parked\|handed-off]` | Close an item |
| `list [--worktree PATH\|--global\|--all]` | List items / presence |
| `checkout` | Mark this worktree presence `closed`; list undischarged this-worktree items (disclosure; exit 0 in v1 unless `--strict` later) |

**`checkin` three-layer print order**

1. **This worktree** — open items for cwd worktree  
2. **Global strip** — items with `severity=blocker` or `owner_scope=juniper` (any worktree), still `open`/`parked`  
3. **Workspace presence** — other worktrees with `status=active` or `stale`: thread summary (paragraph max) + current task; then collision warnings vs cwd

### Presence lifecycle

```text
SessionStart / checkin
    → upsert stub (worktree_path, branch, heartbeat_at, status=active)
    → agent fills thread_summary + current_task via heartbeat
    → if now - heartbeat_at > 30 min → status=stale (still listed, clearly marked)
    → if worktree_path not in `git worktree list` → status=closed
```

**Close paths (defense in depth)**

1. Every `checkin` / SessionStart: reconcile against live `git worktree list` via `worktree_lib`  
2. `scripts/prune_merged_worktrees.py`: after a successful remove, call `agent_board.py` reconcile/close for that path  
3. Explicit `checkout`

### Collision detection (disclosure only)

On `checkin` / SessionStart for cwd:

- Other `active`/`stale` presence rows whose `related_files`, dirty git paths, or inferred `services/<name>` paths overlap the current worktree.
- Graphify community-overlap collision detection is deferred until there is a structured branch/PR mapping for `graphify prs --conflicts`; do not substring-match formatted graphify output.

Print a clear warning block. Do **not** block the session. No `--ack-collision` in v1.

Reuse existing tools; do not invent a collision-type ontology.

### Vendor adapters (thin)

| Vendor | Wire |
|---|---|
| Claude Code | `.claude/settings.json` SessionStart → `python3 scripts/agent_board.py checkin` (in addition to existing worktree summary); Stop → reminder/`checkout` nudge |
| Cursor | New project `.cursor/hooks.json` `sessionStart` / `stop` calling the same CLI |
| Plain terminal | No hooks required |

Failures in hooks: fail open (print nothing / non-blocking), same as `session_start_worktree_summary.py`.

### AGENTS.md pointer

Add a short section (or extend §2): agents working in this repo should `checkin` at session start, `heartbeat` when the current task changes, `add` open decisions/findings instead of only burying them in PR prose, and `checkout` (or rely on worktree-delete reconcile) when done. Priority: track items for the current worktree; elsewhere is fine with `scope_note`.

### Tests (gate)

Deterministic, no network:

- Concurrent append under lock does not corrupt JSONL  
- Worktree missing from fixture `worktree list` → presence auto-closed on reconcile  
- Heartbeat older than 30 min → `stale`  
- `owner_scope != this-worktree` without `scope_note` → rejected  
- Overlapping `related_files` between two presence/item rows → collision warning text present  
- `checkin` print includes global blocker from another worktree  

### Files likely to touch (implementation)

- `scripts/agent_board.py` (new)
- `scripts/agent_board_lib.py` or package under `scripts/` (lock, schema, materialize) — keep thin
- `scripts/hooks/session_start_agent_board.py` (new; Claude adapter)
- `.claude/settings.json` (SessionStart + optional Stop)
- `.cursor/hooks.json` + `.cursor/hooks/` (new)
- `scripts/prune_merged_worktrees.py` (close-on-remove call)
- `scripts/worktree_lib.py` (reuse only; avoid duplication)
- `AGENTS.md` / `CLAUDE.md` (pointer)
- `scripts/tests/test_agent_board.py` or `tests/` location matching sibling script tests
- This spec; later PR report under `docs/superpowers/pr-reports/`

### Non-goals (v1)

- Committing the board into the git repo  
- Scraping PR reports into items  
- graphify export of board nodes  
- Hard or soft gates on collision / undischarged checkout  
- Hub / notify / UI surfaces  
- Keyword or emotional-state detectors (irrelevant; this is agent-ops)

### Acceptance checks

1. Two parallel shells in different worktrees: A `add`s a blocker; B `checkin` sees it in the global strip without any git pull.  
2. B `heartbeat --summary "...' --task "..."`; A `checkin` shows B's summary + task.  
3. Remove B's worktree (or simulate absent from list); next reconcile marks B presence `closed`.  
4. Stop heartbeats >30 min; next `checkin` shows `stale`.  
5. Overlapping `related_files`; `checkin` prints a collision warning and still exits 0.  
6. Claude SessionStart (and Cursor sessionStart once wired) injects the three-layer view.  
7. Plain `python3 scripts/agent_board.py checkin` works with no vendor hooks loaded.

### Risks / concerns

| Severity | Concern | Mitigation |
|---|---|---|
| Medium | Agents forget to fill summary/task → empty stubs | Hooks create stub; SessionStart text nudges; stale marks dead rows |
| Medium | Host-local only → not shared across machines | Accepted for v1; real-time multi-agent on Athena is the priority |
| Low | JSONL grows forever | Periodic compact of closed/resolved older than X days (follow-up, not v1 blocker) |
| Low | Collision false positives | Disclosure only; tune signals after live use |

### Recommended first implementation slice

1. Schema + lock + `add`/`list`/`checkin`/`heartbeat`/`checkout`/`resolve` in `scripts/agent_board.py`  
2. Fixture tests for lock, stale, reconcile-close, global strip, collision warning  
3. Wire Claude SessionStart adapter + prune-script close  
4. Cursor hooks + AGENTS.md pointer  
5. Live smoke: two worktrees, two checkins, prove real-time visibility  

Do **not** start with PR scraping or git export.

## Relationship to existing systems

| System | Relationship |
|---|---|
| Worktree SessionStart summary | Complementary; keep both (counts vs board content) |
| `graphify prs --conflicts` | Manual merge-order risk report; runtime collision integration deferred until output can be mapped structurally to branches/PRs |
| `reviews/pending/` | Unrelated brainstorm parking; do not overload |
| Hub Pending Decisions | Orion cognition; do not overload |
| PR-report Concerns | Still required for shipped PRs; board is the live parking lot so those concerns are not the *only* copy |

## Open follow-ups (explicitly deferred)

- Git export of parked/handed-off items at PR handoff  
- Import of historical "not fixed" PR-report bullets  
- Soft gate (`checkout --strict`) once the board has real usage  
- Compact/GC of closed rows  
- graphify surface for open items
- Structured graphify PR/community collision integration
