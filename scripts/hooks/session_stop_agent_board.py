#!/usr/bin/env python3
"""Stop hook: reminds the agent to checkout or resolve open board items.

Fails silently (no output) on any error -- never blocks session stop.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_board_lib import (  # noqa: E402
    board_config_from_env,
    load_state,
    read_session_id_from_stdin_hook_payload,
    resolve_current_identity,
)

# Ownerless items (no session_id) older than this stop nagging every session
# that ever stops in the worktree -- see _is_stale_ownerless_item below.
_OWNERLESS_ITEM_STALE_AFTER = timedelta(hours=24)

# Where per-session "have I already nagged about exactly this item set"
# markers live. Deliberately separate from the board's own JSONL event log
# (~/.orion/agent-board.jsonl) -- this is Stop-hook bookkeeping, not
# dev-coordination content, so it doesn't belong in the shared event stream
# other tooling (agent_board.py list/checkin, other agents' reads) parses.
# Overridable for tests via env var, same convention as ORION_AGENT_BOARD_PATH.
_NAG_STATE_DIR_ENV = "ORION_AGENT_BOARD_NAG_STATE_DIR"


def _nag_state_dir() -> Path:
    raw = os.environ.get(_NAG_STATE_DIR_ENV)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".orion" / "stop_hook_nag_state"


def _item_set_digest(item_ids: list[str]) -> str:
    joined = "\n".join(sorted(item_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _marker_filename(session_id: str) -> str:
    """Hash session_id into the filename rather than using it raw.

    `session_id` comes from `read_session_id_from_stdin_hook_payload()`,
    which parses arbitrary JSON off stdin with no format validation --
    `resolve_current_identity()`'s own docstring already contemplates this
    hook being invoked by "a non-Claude-Code harness whose Stop payload
    doesn't carry one," i.e. an untrusted-source caller is an acknowledged
    scenario, not hypothetical. A raw `session_id` containing `../` would
    let `_nag_state_dir() / session_id` write/read outside the intended
    directory (path/existence control, not content injection, but still a
    real corruption/DoS primitive). Hashing removes the path-traversal
    surface entirely at negligible cost.
    """
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _already_nagged_this_set(session_id: str, digest: str) -> bool:
    """True if this exact session already emitted a nag for this exact item
    set. Fails open (False, i.e. still nag) on any read error -- matches the
    module's own fail-silent-on-error contract; we'd rather nag once more
    than swallow a real reminder because of a transient disk issue.
    """
    marker = _nag_state_dir() / _marker_filename(session_id)
    try:
        return marker.read_text(encoding="utf-8").strip() == digest
    except OSError:
        return False


def _record_nagged_set(session_id: str, digest: str) -> None:
    """Best-effort write; a failure here should never block emitting the nag
    itself, so this is called after the nag is already queued for output.
    """
    try:
        state_dir = _nag_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        marker = state_dir / _marker_filename(session_id)
        marker.write_text(digest, encoding="utf-8")
        os.chmod(marker, 0o600)
    except OSError:
        pass


def _is_stale_ownerless_item(item: dict) -> bool:
    raw = item.get("updated_at")
    if not raw:
        # No timestamp to judge age by -- keep the old fail-open behavior
        # (treat as fresh, i.e. still nag) rather than guessing.
        return False
    try:
        updated_at = datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - updated_at) > _OWNERLESS_ITEM_STALE_AFTER


def main() -> None:
    if os.environ.get("ORION_FCC_SUBPROCESS"):
        # See matching guard in session_start_agent_board.py: this is a
        # per-turn FCC chat subprocess, not a coding session with a worktree
        # to check out of.
        return
    session_id = read_session_id_from_stdin_hook_payload()
    try:
        cfg = board_config_from_env()
        current = resolve_current_identity(cfg, session_id=session_id)["worktree_path"]
        state = load_state(cfg)
        # Items are worktree-scoped, not session-scoped, so a shared checkout
        # can accumulate open items across many past sessions -- without the
        # session_id check below, a fresh read-only session that added zero
        # items still gets nagged every Stop about work it never touched.
        # Items with no session_id (added before this field existed, or by a
        # plain shell call outside Claude Code) have no session to attribute
        # them to, so a *recent* one still nags rather than going silently
        # unowned. But without a staleness cutoff this becomes permanent: a
        # legacy ownerless item never stops nagging, for every future session
        # that stops in the worktree, since `checkout` closes presence, not
        # items -- live-confirmed 2026-07-24 (dozens of pre-session_id-field
        # findings nagging a session that never touched them). Past the
        # cutoff, treat it as backlog to triage explicitly, not a per-Stop
        # reminder. If *our own* session_id couldn't be resolved
        # (missing/malformed stdin payload, or a non-Claude-Code harness
        # whose Stop payload doesn't carry one), we can't tell "mine" from
        # "someone else's" either -- fail open (fall back to plain worktree
        # scoping, no staleness filter) rather than silently excluding every
        # item that happens to carry a real session_id.
        #
        # Status filter is "open" only, not "open" + "parked" -- confirmed
        # live 2026-07-28: the nag's own remedy text below ("resolve/park
        # them") implied parking was a valid way to quiet it, but the old
        # filter kept nagging on parked items anyway. Parking an item *is*
        # triaging it (a deliberate "not now" decision, distinct from never
        # having looked at it) -- `agent_board.py checkin`'s own listing
        # still surfaces parked items for context, so nothing about parked
        # work goes invisible; it just stops being treated as an unactioned
        # reminder on every single Stop turn.
        open_items = {
            item_id: item
            for item_id, item in state.items.items()
            if item.get("worktree_path") == current
            and item.get("status") == "open"
            and (
                session_id is None
                or item.get("session_id") == session_id
                or (not item.get("session_id") and not _is_stale_ownerless_item(item))
            )
        }
    except Exception:
        return
    if not open_items:
        # Nothing actionable -- saying so on every single Stop is pure token
        # noise, not a reminder. Stay silent, matching the fail-silent
        # pattern already used for errors above.
        return

    # Per-turn dedup: confirmed live 2026-07-28 (7 straight recurrences,
    # documented in project memory) that this hook re-emitted the identical
    # nag on every single Stop turn even when the underlying item set never
    # changed between turns -- no suppression at all previously. Only
    # applies when we have a real session_id to key the marker file by;
    # without one, fall back to the old always-nag behavior (matches the
    # existing fail-open stance for an unresolved session_id above).
    digest = _item_set_digest(list(open_items.keys()))
    if session_id and _already_nagged_this_set(session_id, digest):
        return

    detail = f"Agent board checkout reminder: {len(open_items)} open item(s) remain for this worktree. Run `python3 scripts/agent_board.py checkout` or resolve/park them."
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": detail,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    if session_id:
        _record_nagged_set(session_id, digest)


if __name__ == "__main__":
    main()
