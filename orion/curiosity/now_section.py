"""Curiosity reads Orion's now: the kickoff section and the cites in the turn text (integration patch 3). Pure.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A ("Who reads the now":
"Curiosity candidate selection. A painting or a prediction surprise can become a curiosity candidate with a
cite"). Hub reads ``GET /temporal-self/now?organ=curiosity`` at kickoff (``services/orion-hub/scripts/
curiosity_now.py``); this module renders what it was shown and finds the ``[experience:<id>]`` cites in the
turn's text at the turn's end. The cites are checked against the shown row by orion-durable-runs
(``orion.temporal_self.now.record_use``), never here: a cite of an id that was not shown is sent and counted
there as ``rejected_not_shown``, so an invented cite is measured, not hidden.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

from orion.schemas.temporal_self import NOW_MAX_CITES, TemporalNowV1
from orion.temporal_self.now import EXPERIENCE_PREFIX

# A cite as Orion writes it in the turn text. Event ids carry colons, dots and dashes
# (``office_session:cam0-tr-0``), so everything up to the closing bracket is the id. Angle brackets never
# belong to an id: the prompt's own example ``[experience:<id>]`` is not a cite.
CITE_RE = re.compile(r"\[\s*" + re.escape(EXPERIENCE_PREFIX) + r"\s*([^\]\s<>]+)\s*\]")

HEADER = "WHAT YOUR OTHER ORGANS LIVED SINCE YOUR LAST RUN"


def parse_now(raw: Any) -> Optional[TemporalNowV1]:
    """The endpoint's body as ``TemporalNowV1``, or None when it is not one (never raises)."""
    if isinstance(raw, TemporalNowV1):
        return raw
    if not isinstance(raw, Mapping):
        return None
    try:
        return TemporalNowV1.model_validate(raw)
    except Exception:  # noqa: BLE001 -- a malformed body is no section, never a failed kickoff
        return None


def format_now_section(now: Optional[TemporalNowV1]) -> list[str]:
    """The kickoff block: each item with its cite tag, organ, code-written label and salience. Nothing at
    all when there is no now (read off, failed, malformed) or it holds no item."""
    if now is None or not now.items:
        return []
    lines = [
        f"{HEADER} (Orion's now: recent experiences of your other organs, read when this run started, one per "
        "organ before any organ's second. Salience is how unusual the event was for its own source over 30 "
        "days, 0 to 1; 'unknown' means too little history or no measure for that source).",
        "",
    ]
    for item in now.items:
        sal = "unknown" if item.salience is None else f"{item.salience:.2f}"
        lines.append(f"  - [{EXPERIENCE_PREFIX}{item.event_id}] {item.organ}, "
                     f"{item.occurred_at:%Y-%m-%d %H:%MZ}: {item.label} (salience {sal})")
    lines += [
        "",
        "Any of these can become what you look into: a painting, a dream, a prediction that surprised you, "
        "something a camera saw. If one of them shaped what you chose or what you found, cite it in your "
        f"final reply exactly as shown, e.g. [{EXPERIENCE_PREFIX}<id>]. At most {NOW_MAX_CITES} count, and "
        "only ids listed here. Citing none is a real answer too.",
        "",
    ]
    return lines


def extract_cites(text: Optional[str], cap: int = NOW_MAX_CITES, shown: Optional[Iterable[str]] = None) -> list[str]:
    """The distinct ``[experience:<id>]`` ids in ``text``, at most ``cap``, in order of appearance. With
    ``shown`` (the event ids the run was shown), shown ids go first, so an invented cite never pushes a real
    one out of the cap; invented ones still fill any room left, so the endpoint counts them."""
    cap = max(0, int(cap))
    found: list[str] = []
    for m in CITE_RE.finditer(text or ""):
        eid = m.group(1).strip()
        if eid and eid not in found:
            found.append(eid)
    if shown is not None:
        known = set(shown)
        found = [c for c in found if c in known] + [c for c in found if c not in known]
    return found[:cap]


__all__ = ["CITE_RE", "HEADER", "extract_cites", "format_now_section", "parse_now"]
