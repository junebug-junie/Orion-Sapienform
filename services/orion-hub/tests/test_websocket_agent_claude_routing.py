"""WebSocket agent-claude mode must branch to FCC bridge, not context-exec."""
from __future__ import annotations

from pathlib import Path


HUB_ROOT = Path(__file__).resolve().parents[1]
WS_PATH = HUB_ROOT / "scripts" / "websocket_handler.py"


def test_websocket_handler_imports_agent_claude_bridge() -> None:
    source = WS_PATH.read_text(encoding="utf-8")
    assert "run_turn_from_settings" in source
    assert "prepare_agent_claude_input" in source
    assert 'mode == "agent-claude"' in source or "agent-claude" in source


def test_websocket_handler_emits_claude_step_kind() -> None:
    source = WS_PATH.read_text(encoding="utf-8")
    assert '"claude_step"' in source or "'claude_step'" in source


def test_turn_orchestrator_import_is_guarded_and_reports_a_client_facing_error() -> None:
    """Regression guard for the 2026-08-23 production outage: `from
    orion.hub.turn_orchestrator import run_unified_turn` used to be a bare
    module-level import inside the per-message loop -- when it raised
    (a Python-3.11-only `datetime.UTC` import in a transitive dependency,
    on this container's actual Python 3.10 runtime), the exception
    propagated past this whole handler with no client-facing error frame,
    silently killing the WebSocket connection. Every orion-mode chat turn
    was down; from the browser it looked exactly like a hang, not a crash.

    No live-execution test exists for this file (it has no FastAPI
    TestClient WebSocket harness -- see test_websocket_handler_imports_
    agent_claude_bridge above for the same static-source-check convention
    already established here), so this checks the actual guard shape:
    guarded, sent via the disconnect-safe helper, and logged server-side."""
    source = WS_PATH.read_text(encoding="utf-8")
    import_stmt = "from orion.hub.turn_orchestrator import run_unified_turn"
    assert import_stmt in source

    idx = source.index(import_stmt)
    # The import must sit inside a try block, not be a bare statement --
    # look at what immediately precedes it.
    preceding = source[max(0, idx - 200) : idx]
    assert "try:" in preceding

    # And the except branch right after it must (a) log server-side, (b)
    # send a client-facing turn_error frame via the disconnect-safe
    # helper, and (c) not swallow the just-appended-but-never-answered
    # user turn into `history`.
    following = source[idx : idx + 800]
    assert "except ImportError" in following
    assert "logger.error" in following
    assert '"phase": "import"' in following
    assert "_safe_ws_send_json" in following
    assert "history.pop()" in following


def test_unified_path_emits_turn_started_after_correlation_id() -> None:
    """Mid-turn Cockpit needs the client to learn correlation_id before reply.

    Static guard (same convention as other websocket_handler checks here):
    right after active_turn correlation_id is set on the unified path, Hub
    must fail-open send a turn_started frame via _safe_ws_send_json.
    """
    source = WS_PATH.read_text(encoding="utf-8")
    marker = 'active_turn["correlation_id"] = trace_id'
    # Unified Orion/agent path is the first assignment after the import guard.
    idx = source.index(marker)
    following = source[idx : idx + 900]
    assert '"kind": "turn_started"' in following
    assert '"correlation_id": trace_id' in following
    assert '"mode": client_mode' in following
    assert "_safe_ws_send_json" in following
    # turn_started must land before run_unified_turn is invoked.
    run_idx = source.index("run_unified_turn(", idx + len(marker))
    started_idx = source.index('"kind": "turn_started"', idx)
    assert started_idx < run_idx


def test_reply_stamp_is_computed_before_the_lane_split_and_reaches_both_lanes() -> None:
    """Reply stamp (2026-09-22). Review of the first cut found it stamped
    only the legacy lane's `turn_client_meta`; the unified lane (the live
    default for every one of Juniper's turns, `ORION_UNIFIED_TURN_ENABLED=true`)
    `continue`s before that dict is built and published its rows with no
    client_meta at all. So: computed ONCE before `if client_mode in
    ("orion", "agent")`, handed to `run_unified_turn(client_meta=...)`, and
    merged into `turn_client_meta` for the legacy lane. Static check, same
    convention as the rest of this file; the orchestrator side is covered
    behaviourally in test_chat_history_no_raw_publish.py."""
    source = WS_PATH.read_text(encoding="utf-8")
    assert "from scripts.outreach_provenance import reply_stamp_for_session" in source
    stamp = source.index("reply_stamp = await reply_stamp_for_session(session_id)")
    split = source.index('if client_mode in ("orion", "agent") and settings.ORION_UNIFIED_TURN_ENABLED:')
    unified = source.index("client_meta=reply_stamp or None,")
    build = source.index("turn_client_meta = dict(client_meta)")
    legacy = source.index("turn_client_meta.update(reply_stamp)")
    publish = source.index('_schedule_publish(publish_chat_history(bus, [user_env]), "chat.history user")')
    assert stamp < split < unified < build < legacy < publish
    # The unified call actually receives it.
    run_call = source.index("run_unified_turn(\n                            websocket,")
    assert run_call < unified < source.index("continue", unified)
    # Gated the same way the history publish is, so a no_write turn never
    # spends a lookup it cannot use.
    assert "if bus and not no_write:\n                reply_stamp = await reply_stamp_for_session(session_id)" in source
    # Exactly one legacy dict is stamped and every later publish derives from it.
    assert source.count("enriched_client_meta = dict(turn_client_meta)") == 1
