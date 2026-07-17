# PR report: agent workspace board

## Summary

- Added host-local `~/.orion/agent-board.jsonl` workspace board with `fcntl` locking, owner-only file modes, and append-only JSONL events.
- Added vendor-neutral CLI (`scripts/agent_board.py`) for `checkin`, `heartbeat`, `add`, `resolve`, `list`, `checkout`, and `reconcile`.
- Added Claude/Cursor SessionStart/Stop hook adapters that fail open.
- Wired prune close-on-remove so deleted worktrees close board presence.
- Added AGENTS/CLAUDE guidance and focused script tests.

## Outcome moved

Concurrent agents can see this-worktree items, global blockers/Juniper escalations, other active/stale worktrees, and disclosure-only collision warnings without git round-trips. Collision signals cover explicit `--files`, dirty git paths, and shared `services/<name>` paths. Live smoke: second worktree `checkin` saw the first worktree's global blocker and presence summary in real time via a shared host board file.

## Current architecture

Before this patch, worktree SessionStart only surfaced counts, PR reports buried follow-ups in prose, and no host-shared real-time agent board existed.

## Architecture touched

Scripts and hooks only: `scripts/agent_board*.py`, `scripts/hooks/session_*_agent_board.py`, `.claude/settings.json`, `.cursor/hooks.json`, `scripts/prune_merged_worktrees.py`, and AGENTS/CLAUDE guidance. Spec + plan under `docs/superpowers/`.

## Files changed

- `scripts/agent_board_lib.py`: board storage, validation, reconciliation, collision detection, rendering
- `scripts/agent_board.py`: vendor-neutral CLI
- `scripts/hooks/session_start_agent_board.py`: SessionStart adapter
- `scripts/hooks/session_stop_agent_board.py`: Stop adapter
- `.claude/settings.json`: Claude hook wiring
- `.cursor/hooks.json`: Cursor hook wiring
- `scripts/prune_merged_worktrees.py`: close presence when removing worktrees
- `CLAUDE.md`: operator/agent guidance
- `tests/scripts/*agent_board*`: deterministic tests
- `docs/superpowers/specs/2026-07-16-agent-workspace-board-design.md`: design
- `docs/superpowers/plans/2026-07-16-agent-workspace-board.md`: implementation plan

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: local hook context now includes agent board state
- Compatibility notes: board file is host-local and outside git; override with `ORION_AGENT_BOARD_PATH` for tests/smoke

## Env/config changes

- Added keys: none (optional runtime override `ORION_AGENT_BOARD_PATH` for isolation only; not an operator `.env` contract)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
/mnt/scripts/Orion-Sapienform/orion_dev/bin/pytest \
  tests/scripts/test_agent_board_lib.py \
  tests/scripts/test_agent_board_cli.py \
  tests/scripts/test_session_agent_board_hooks.py \
  tests/scripts/test_prune_merged_worktrees.py \
  tests/scripts/test_session_start_worktree_summary.py \
  tests/scripts/test_worktree_lib.py \
  -q
47 passed in 16.17s
```

Note: pytest-asyncio prints a deprecation warning about unset `asyncio_default_fixture_loop_scope` (pre-existing environment noise, unrelated to this patch).

## Evals run

```text
No eval harness applies; this is agent-ops tooling, not Orion cognition behavior.
```

## Docker/build/smoke checks

```text
python3 -m py_compile scripts/agent_board.py scripts/agent_board_lib.py \
  scripts/hooks/session_start_agent_board.py scripts/hooks/session_stop_agent_board.py
python3 -m json.tool .claude/settings.json
python3 -m json.tool .cursor/hooks.json
COMPILE_JSON_OK

ORION_AGENT_BOARD_PATH=/tmp/orion-agent-board-smoke.jsonl
# From docs/agent-workspace-board worktree:
python3 scripts/agent_board.py heartbeat --summary "Agent A smoke summary." --task "Writing board smoke."
python3 scripts/agent_board.py add --kind blocker --severity blocker --scope juniper \
  --scope-note "Smoke test global item." --summary "Smoke blocker visible globally." \
  --files scripts/agent_board.py
python3 scripts/agent_board.py checkin
# PASS: This worktree + Global strip + smoke blocker

# From shared checkout (/mnt/scripts/Orion-Sapienform) same board path:
python3 .../scripts/agent_board.py checkin
# PASS: Global strip shows smoke blocker; Workspace presence shows Agent A summary/task

scripts/safe_graphify_update.sh
# REFUSED: node count dropped ~92% (known destructive-update failure mode).
# Restored graphify-out; nothing to commit from graphify.
No Docker restart required.
```

## Review findings fixed

- Finding: pre-merge review found `os.chmod` on the board parent raised `PermissionError` when `ORION_AGENT_BOARD_PATH` pointed at a file under shared parents like `/tmp`, breaking smoke isolation and any non-owned parent path.
  - Fix: `_chmod_best_effort` fails soft on `OSError`; owned `.orion` dirs still get `0700`/`0600`.
  - Evidence: `test_append_event_tolerates_unwritable_shared_parent`.
- Finding: final review found `checkout` counted every open/parked item on the host board, not only the current worktree.
  - Fix: checkout now filters open items to `current_worktree_identity()["worktree_path"]`.
  - Evidence: `test_checkout_counts_only_this_worktree_open_items`.
- Finding: final review found `resolve` accepted unknown IDs and reported success even though replay ignored the orphan status event.
  - Fix: `change_item_status` validates the item exists before appending `item_status_changed`.
  - Evidence: `test_resolve_rejects_unknown_item_id`.
- Finding: final review found one malformed JSONL line made the board unreadable.
  - Fix: `_read_events` skips malformed/incomplete lines and append flushes/fsyncs each event.
  - Evidence: `test_load_state_skips_corrupt_jsonl_lines`.
- Finding: final review found board directories/files defaulted to world-readable permissions.
  - Fix: board directory is forced to `0700`; board and lock files are forced to `0600`.
  - Evidence: `test_append_event_creates_private_board_file_and_directory`.
- Finding: final review found prune close-on-remove used unresolved paths while presence keys are resolved.
  - Fix: prune passes `str(w.path.resolve())`; `close_presence` also normalizes explicit paths.
  - Evidence: updated prune close test asserts the resolved path.
- Finding: final review found collision detection narrowed to explicit `--files`, missing design-level dirty/service signals and making an unsafe graphify substring signal tempting.
  - Fix: collision detection now includes dirty git paths and shared `services/<name>` paths; graphify community overlap is explicitly deferred until branch/PR mapping is structured.
  - Evidence: `test_detect_collisions_reports_same_service_path`; dirty paths are fail-open and covered by checkin behavior.
- Finding: post-fix review found the first graphify branch-overlap implementation was substring-based against formatted output and could false-positive.
  - Fix: removed runtime graphify collision matching and updated design/report/docs to state graphify integration is deferred.
  - Evidence: focused suite passes with no graphify runtime collision dependency.
- Finding: `reconcile_closed_worktrees` never persisted `presence_closed` (load-with-live-paths first made the append loop dead).
  - Fix: load persisted state without `live_worktrees`, append closes, then reload with live paths (`830dd979`).
  - Evidence: strengthened `test_reconcile_closed_worktrees_persists_closed_event` asserts JSONL event + closed status without live set; 16/16 board tests pass.
- Finding (Task 1 Important, deferred then satisfied): heartbeat must refresh `heartbeat_at`.
  - Fix: Task 2 `upsert_presence` always sets `heartbeat_at` in payload.
  - Evidence: Task 2 review Approved.
- Whole-branch review: completed after PR creation; Critical none; Important findings above fixed in follow-up commit.

## Restart required

```text
No restart required.
New Claude/Cursor sessions will pick up SessionStart/Stop hooks automatically.
Existing sessions need a new session to load updated hook settings.
```

## Risks / concerns

- Severity: Low
- Concern: Host-local board does not sync across machines.
- Mitigation: Accepted v1 tradeoff for real-time multi-agent visibility on Athena; git export deferred.

- Severity: Low
- Concern: `scripts/safe_graphify_update.sh` refused (~92% node drop); graph not updated for new scripts.
- Mitigation: Documented; full re-extraction may be needed later; not blocking agent-board functionality.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1100
