from __future__ import annotations

from typing import Any


def load_session_presence(session_id: str | None, store: Any | None) -> dict[str, Any] | None:
    """Return session-scoped audience presence, or None when unavailable."""
    if not store or not session_id:
        return None
    payload = store.get(str(session_id))
    return dict(payload) if isinstance(payload, dict) else None


def _payload_has_presence_context(payload: dict[str, Any]) -> bool:
    presence = payload.get("presence_context")
    if not isinstance(presence, dict) or not presence:
        return False
    if presence.get("audience_mode"):
        return True
    companions = presence.get("companions")
    return isinstance(companions, list) and bool(companions)


def inject_session_presence(payload: dict[str, Any], session_id: str | None, store: Any | None) -> dict[str, Any]:
    """Merge stored presence into a chat payload when the client did not send one."""
    routed = dict(payload or {})
    if _payload_has_presence_context(routed):
        return routed
    stored = load_session_presence(session_id, store)
    if stored:
        routed["presence_context"] = stored
    return routed
