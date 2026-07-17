# PR report: agent workspace board

## Summary

- Added host-local `~/.orion/agent-board.jsonl` workspace board with `fcntl` locking and append-only JSONL events.
- Added vendor-neutral CLI (`scripts/agent_board.py`) for `checkin`, `heartbeat`, `add`, `resolve`, `list`, `checkout`, and `reconcile`.
- Added Claude/Cursor SessionStart/Stop hook adapters that fail open.
- Wired prune close-on-remove so deleted worktrees close board presence.
- Added AGENTS/CLAUDE guidance and focused script tests.

## Outcome moved

Concurrent agents can see this-worktree items, global blockers/Juniper escalations, other active/stale worktrees, and disclosure-only collision warnings without git round-trips. Live smoke: second worktree `checkin` saw the first worktree's global blocker and presence summary in real time via a shared host board file.

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
41 passed in 13.08s
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

- Finding: `reconcile_closed_worktrees` never persisted `presence_closed` (load-with-live-paths first made the append loop dead).
  - Fix: load persisted state without `live_worktrees`, append closes, then reload with live paths (`830dd979`).
  - Evidence: strengthened `test_reconcile_closed_worktrees_persists_closed_event` asserts JSONL event + closed status without live set; 16/16 board tests pass.
- Finding (Task 1 Important, deferred then satisfied): heartbeat must refresh `heartbeat_at`.
  - Fix: Task 2 `upsert_presence` always sets `heartbeat_at` in payload.
  - Evidence: Task 2 review Approved.
- Nested whole-branch review: deferred to orchestrator final review after this report.

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

- Severity: Low
- Concern: Optional path resolve mismatch between prune `w.path` and identity `.resolve()` on symlink-heavy hosts.
- Mitigation: Noted in Task 5 review; follow-up if observed live.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1100
