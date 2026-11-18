"""Sealed contractor prompt + lenient PeerBrief parse for the Claude helper.

Pure functions: no bus, no subprocess. The worker sends the prompt to the
room companion (Claude) and maps the reply text back with
``parse_peer_brief_body``.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from orion.curiosity.peer_briefs import strip_self_definition_draft
from orion.schemas.curiosity_peer import (
    CuriosityPeerNameV1,
    HelpRequestV1,
    PeerBriefV1,
)

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def build_sealed_prompt(
    help_req: HelpRequestV1,
    *,
    context_pack: str = "",
) -> str:
    """Build the sealed contractor prompt (mode rules + PeerBrief JSON shape)."""
    mode = help_req.mode
    lines = [
        "You are a read-only contractor peer hired by Orion.",
        "Use only read-only tools you are given. Do not edit, shell, delete, or mutate.",
        "Do not write :Prior, :Finding, :PeerBrief, or belief graph nodes.",
        f"Mode: {mode}.",
        "",
    ]
    if mode == "self_inquiry":
        lines += [
            "Self-inquiry mode rules:",
            "- Point at evidence only (paths, queries, hop notes, ledger refs).",
            "- Never draft :SelfDefinition text, MERGE (s:SelfDefinition ...),",
            "  or identity prose such as 'I am a digital mind'.",
            "- Orion alone writes SelfDefinition.",
            "",
        ]
    else:
        lines += [
            "World-curiosity mode rules:",
            "- Return investigative notes Orion can use or ignore.",
            "- Do not author priors or findings yourself.",
            "",
        ]

    lines += [
        f"help_id: {help_req.help_id}",
        f"run_id: {help_req.run_id}",
        f"prior_id: {help_req.prior_id or ''}",
        f"question: {help_req.question}",
        f"tried_summary: {help_req.tried_summary}",
        f"success_criteria: {help_req.success_criteria}",
        "",
    ]
    if context_pack.strip():
        lines += ["Sealed context pack:", context_pack.strip(), ""]

    lines += [
        "Respond with a single JSON object (no markdown fence required) with keys:",
        '  "summary": string,',
        '  "evidence_pointers": string[],',
        '  "open_questions": string[],',
        '  "suggested_next_looks": string[]',
        "Keep summary concrete and evidence-linked. Empty useful content is allowed;",
        "prefer an honest empty summary over invented findings.",
    ]
    return "\n".join(lines)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    match = _JSON_OBJECT_RE.search(raw)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    out: list[str] = []
    for item in list(value):
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _strip_self_def_list(items: list[str]) -> list[str]:
    """Drop or clean SelfDefinition-shaped entries from peer list fields."""
    cleaned: list[str] = []
    for item in items:
        text, _stripped = strip_self_definition_draft(item)
        if text.strip():
            cleaned.append(text.strip())
    return cleaned


def parse_peer_brief_body(
    body: str,
    *,
    help: HelpRequestV1,
    peer: CuriosityPeerNameV1 = "claude_room",
) -> PeerBriefV1:
    """Leniently map the helper's reply text into PeerBriefV1.

    Empty body → status=empty. Self-inquiry strips SelfDefinition drafts from
    summary and list fields; if that empties useful content → status=empty.
    """
    brief_id = f"brief-{uuid.uuid4().hex[:12]}"
    base = dict(
        brief_id=brief_id,
        help_id=help.help_id,
        run_id=help.run_id,
        prior_id=help.prior_id,
        peer=peer,
    )

    raw = (body or "").strip()
    if not raw:
        return PeerBriefV1(**base, status="empty", summary="")

    data = _extract_json_object(raw)
    if data is None:
        summary = raw
        pointers: list[str] = []
        opens: list[str] = []
        looks: list[str] = []
    else:
        summary = str(data.get("summary") or "").strip()
        pointers = _as_str_list(data.get("evidence_pointers"))
        opens = _as_str_list(data.get("open_questions"))
        looks = _as_str_list(data.get("suggested_next_looks"))

    if help.mode == "self_inquiry":
        summary, _stripped = strip_self_definition_draft(summary)
        pointers = _strip_self_def_list(pointers)
        opens = _strip_self_def_list(opens)
        looks = _strip_self_def_list(looks)

    useful = bool(summary.strip() or pointers or opens or looks)
    if not useful:
        return PeerBriefV1(
            **base,
            status="empty",
            summary=summary,
            evidence_pointers=pointers,
            open_questions=opens,
            suggested_next_looks=looks,
        )

    return PeerBriefV1(
        **base,
        status="ok",
        summary=summary,
        evidence_pointers=pointers,
        open_questions=opens,
        suggested_next_looks=looks,
    )
