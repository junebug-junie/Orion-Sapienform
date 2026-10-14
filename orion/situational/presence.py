"""One normalizer for Hub presence (who is in the room with Juniper).

Presence is typed by hand in the Hub. Before this module, nothing checked it
on save, and the turn-time parser fed it straight into strict
``PresenceContextV1``. One stray value -- live 2026-10-14 (correlation
21c8a610): a companion saved with relationship="Pip", age_band="Pip" --
raised a ValidationError that threw away the WHOLE situation brief, so Orion
told a bedtime story to a 7-year-old without knowing a child was listening.

Every reader goes through ``normalize_presence_context``: Hub's
``POST /api/presence`` (so the stored copy is clean), ``_presence_from_ctx``
and the situation cache fingerprint (so both agree byte-for-byte). A bad enum
value falls back to that field's schema default instead of failing the turn.
"""

from __future__ import annotations

from typing import Any, get_args

from orion.schemas.situation import PresenceCompanionV1, PresenceContextV1, RequestorContextV1

MAX_COMPANIONS = 8
_NAME_MAX_CHARS = 60
_NOTE_MAX_CHARS = 120


def _allowed(model: type, field: str) -> tuple[str, ...]:
    return tuple(get_args(model.model_fields[field].annotation))


def _choice(model: type, field: str, value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in _allowed(model, field) else model.model_fields[field].default


def _clean_text(value: Any, cap: int) -> str | None:
    # Names and notes reach the prompt verbatim: one line, bounded.
    text = " ".join(str(value or "").split())
    return text[:cap] or None


def _companion(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = _clean_text(item.get("display_name"), _NAME_MAX_CHARS)
    if not name:
        return None
    relationship = _choice(PresenceCompanionV1, "relationship", item.get("relationship"))
    age_band = _choice(PresenceCompanionV1, "age_band", item.get("age_band"))
    if relationship == "child" and age_band == "unknown":
        age_band = "child"
    return {
        "display_name": name,
        "relationship": relationship,
        "role": _choice(PresenceCompanionV1, "role", item.get("role")),
        "age_band": age_band,
        "context_note": _clean_text(item.get("context_note"), _NOTE_MAX_CHARS),
        "safety_notes": _clean_text(item.get("safety_notes"), _NOTE_MAX_CHARS),
    }


def is_child(companion: dict[str, Any]) -> bool:
    return companion.get("age_band") == "child" or companion.get("relationship") == "child"


def _derived_audience_mode(companions: list[dict[str, Any]]) -> str:
    """The companions are the truth; the mode is a summary of them."""
    kids = [c for c in companions if is_child(c)]
    others = [c for c in companions if not is_child(c)]
    if kids:
        return "family" if others else "kid_present"
    if all(c["relationship"] == "spouse" for c in others):
        return "spouse_present"
    if any(c["relationship"] == "guest" for c in others):
        return "guest_present"
    return "mixed_group"


def normalize_presence_context(raw: Any, *, default_requestor: str) -> dict[str, Any]:
    """Return a presence dict that always validates as ``PresenceContextV1``.

    Timestamps and persist gating stay with the caller; everything else is
    coerced here. With companions, ``audience_mode`` is derived from them so
    the two can never contradict; without, a valid given mode is kept. That
    means a caller cannot send companions plus a mode like "operator_review":
    the people present decide the audience.
    """
    raw = raw if isinstance(raw, dict) else {}
    req = raw.get("requestor") if isinstance(raw.get("requestor"), dict) else {}
    companions_raw = raw.get("companions") if isinstance(raw.get("companions"), list) else []
    companions = [c for c in (_companion(item) for item in companions_raw[:MAX_COMPANIONS]) if c]
    if companions:
        audience_mode = _derived_audience_mode(companions)
    else:
        given = str(raw.get("audience_mode") or "").strip().lower()
        audience_mode = given if given in _allowed(PresenceContextV1, "audience_mode") else "solo"
    return {
        "kind": "presence.context.v1",
        "requestor": {
            "display_name": _clean_text(req.get("display_name"), _NAME_MAX_CHARS) or default_requestor,
            "relationship_to_orion": str(req.get("relationship_to_orion") or "primary_operator"),
            "source": str(req.get("source") or "hub_manual"),
            "confidence": _choice(RequestorContextV1, "confidence", req.get("confidence")),
        },
        "companions": companions,
        "audience_mode": audience_mode,
        "source": _choice(PresenceContextV1, "source", raw.get("source")),
        "persist_to_memory": bool(raw.get("persist_to_memory", False)),
        "privacy_mode": _choice(PresenceContextV1, "privacy_mode", raw.get("privacy_mode")),
        "notes": _clean_text(raw.get("notes"), _NOTE_MAX_CHARS),
    }


_PHRASE_MAX_PEOPLE = 4
_PHRASE_NOTE_CHARS = 40
_PHRASE_MAX_CHARS = 240


def companions_phrase(companions: list[Any]) -> str:
    """'Pip (child, listener)' style list for the prompt; unknowns omitted.

    Bounded (4 people, 40-char notes, 240 chars total) because it shares the situation
    fragment's char cap with everything else in the body.
    """
    parts = []
    for c in companions[:_PHRASE_MAX_PEOPLE]:
        d = c if isinstance(c, dict) else c.model_dump()
        tags = [d.get("relationship"), d.get("role")]
        if d.get("age_band") not in ("unknown", None) and d.get("age_band") != d.get("relationship"):
            tags.insert(1, f"age {d['age_band']}")
        tags = [t for t in tags if t and t not in ("other", "nearby")]
        note = f"; {d['context_note'][:_PHRASE_NOTE_CHARS]}" if d.get("context_note") else ""
        parts.append(f"{d['display_name']} ({', '.join(tags)}{note})" if tags or note else d["display_name"])
    extra = len(companions) - _PHRASE_MAX_PEOPLE
    phrase = ", ".join(parts) + (f", and {extra} more" if extra > 0 else "")
    return phrase if len(phrase) <= _PHRASE_MAX_CHARS else phrase[: _PHRASE_MAX_CHARS - 1] + "…"
