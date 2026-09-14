"""Cursor Agent CLI invoker — sealed prompt, ask-mode argv, PeerBrief parse.

Same idea as FCC / ``claude -p``: subprocess argv, print mode, host/desktop
auth. Never import ``cursor_sdk``. Tests inject ``subprocess.run``.
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from typing import Any, Callable, Optional

from app.cursor_errors import TokenUnavailable, classify_cursor_failure
from app.policy import assert_read_only_cli_argv, build_cursor_agent_argv
from orion.curiosity.peer_briefs import strip_self_definition_draft
from orion.schemas.curiosity_peer import (
    CuriosityPeerNameV1,
    HelpRequestV1,
    PeerBriefV1,
)

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")

# Injectable for tests (never call a live agent from pytest).
SubprocessRun = Callable[..., subprocess.CompletedProcess[str]]


def build_sealed_prompt(
    help_req: HelpRequestV1,
    *,
    context_pack: str = "",
) -> str:
    """Build the sealed contractor prompt (mode rules + PeerBrief JSON shape)."""
    mode = help_req.mode
    lines = [
        "You are a read-only contractor peer hired by Orion.",
        "Investigate with read/grep/glob/ls only. Do not edit, shell, delete, or mutate.",
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
    peer: CuriosityPeerNameV1 = "cursor_auto",
) -> PeerBriefV1:
    """Leniently map Cursor/Claude text into PeerBriefV1.

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


def _combined_cli_output(proc: subprocess.CompletedProcess[str]) -> str:
    parts = [proc.stdout or "", proc.stderr or ""]
    return "\n".join(p for p in parts if p).strip()


def run_cursor_job(
    help: HelpRequestV1,
    sealed_prompt: str,
    *,
    agent_bin: str,
    cwd: str,
    model: str,
    timeout_sec: float = 600.0,
    run: Optional[SubprocessRun] = None,
) -> PeerBriefV1:
    """Spawn Cursor Agent CLI in ask/print mode; map stdout to PeerBriefV1."""
    argv = build_cursor_agent_argv(
        agent_bin=agent_bin,
        prompt=sealed_prompt,
        workspace=cwd,
        model=model,
    )
    assert_read_only_cli_argv(argv)

    runner = run or subprocess.run
    try:
        proc = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TokenUnavailable(f"cursor agent binary not found: {agent_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"cursor agent timed out after {exc.timeout}s") from exc

    combined = _combined_cli_output(proc)
    if proc.returncode != 0:
        err = RuntimeError(
            f"cursor agent exited {proc.returncode}: {combined[:2000]}"
        )
        if classify_cursor_failure(err) == "token_unavailable":
            raise TokenUnavailable(str(err)) from err
        raise err

    return parse_peer_brief_body(combined, help=help)
