"""Runtime policy for Graph autonomy multi-subject SPARQL fan-out (bounded vs full)."""

from __future__ import annotations

from typing import Any, Literal, Mapping

AutonomySubjectFanout = Literal["bounded", "full"]


def autonomy_subject_fanout_from_runtime_ctx(ctx: Mapping[str, Any] | None) -> AutonomySubjectFanout:
    """Return ``bounded`` only for the default Hub quick lane; deep lanes use ``full``.

    ``bounded`` pairs with repository short-circuiting (first preferred subject wins)
    when ``AUTONOMY_CHAT_STANCE_SHORT_CIRCUIT`` and consumer allow lists permit it.

    ``full`` waits for every subject (orion, relationship, juniper) — used for agent
    runtime, brain / general chat, ``chat_quick`` with ``chat_quick_full_stance``, and
    any unknown or missing context (safe default).
    """
    if not ctx:
        return "full"
    mode = str(ctx.get("mode") or "").strip().lower()
    if mode == "agent":
        return "full"
    verb = str(ctx.get("verb") or "").strip().lower()
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    hub_full = bool(opts.get("chat_quick_full_stance"))
    if verb == "chat_quick" and not hub_full:
        return "bounded"
    return "full"
