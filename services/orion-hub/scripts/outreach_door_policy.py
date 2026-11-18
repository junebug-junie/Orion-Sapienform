"""Which composed messages may reach Juniper: one door policy for every outreach door.

Juniper, 2026-11-10: about 17 messages a day reached her and most read as plumbing. Her rules:

- Self-inquiry results ("how Orion thinks about themselves") reach her, even when abstract.
- Investigations of Orion's own code and systems do not, unless she asked for that run.
- The background tick follows the same rule: it does not relay system material.
- A finding she already got is not sent again, by any door.
- Every composed message says, in a short structured line, why it is for her. No line, no send.

Every decision here is deterministic and reads a field that already exists on the record. No word
lists over the message text (repo rule: do not patch misses by growing word lists).

Curiosity (``curiosity_reaches_juniper``): the run's ``line`` decides. ``self_inquiry`` passes.
Any other line (``investigate``, ``self_sense_eval``, unknown) passes only when the run took a
curiosity focus (``curiosity_focus_request``) whose ``origin`` is ``juniper`` and which carries the
chat turn it came from (``chat_correlation_id``). ``requested_by=juniper`` on a read or a tool call
is NOT this signal: every chat-turn tool call carries it, Orion's own included (#2913).

Tick (``endogenous_outreach._fetch_recent_turns``, ``daydream_content_id``): the tick's material is
the daydream, the tension trigger and the real conversation. Orion's own unsolicited messages
(``chat_history_log.client_meta.unsolicited``) are not material: Juniper already got them, and
re-reading them is how the tick sent the same finding three times on 2026-11-09 (15:05, 16:03,
16:53 UTC). A daydream a sent tick already used is not used again (7-day ledger).

Why Juniper (``orion.hub.why_juniper.split_why_juniper``): the composer ends its message with one
line ``Why Juniper: <reason>``. We strip it from the delivered text and record it. Empty or missing
means the message is not sent (``reason=no_why_juniper``). We only check that the field is there;
we do not judge it with a second model.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Tuple

from orion.hub.why_juniper import WHY_JUNIPER_INSTRUCTION, split_why_juniper  # noqa: F401 -- re-export

# Curiosity lines (orion/curiosity/self_inquiry.py LINE_SELF_INQUIRY). Only this one reaches
# Juniper unasked.
CURIOSITY_LINES_FOR_JUNIPER = frozenset({"self_inquiry"})
FOCUS_ORIGIN_JUNIPER = "juniper"

# Decision-row reasons this module produces (``endogenous_outreach_decisions.reason``).
NOT_FOR_JUNIPER_REASON = "not_for_juniper"
NO_WHY_JUNIPER_REASON = "no_why_juniper"
ALREADY_TOLD_REASON = "already_told"

# A curiosity run sends at most one message; its key is the run id. Run ids are unique, so any
# window is correct; 30 days only bounds the scan (a run reaches out within hours of starting).
ALREADY_TOLD_LOOKBACK_DAYS = 30


def curiosity_reaches_juniper(
    line: str,
    *,
    focus_origin: Optional[str] = None,
    focus_chat_correlation_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """(may_send, why) for a curiosity run's reach-out. Pure.

    ``why`` is a short machine label recorded on the decision row:
    ``line:self_inquiry``, ``asked_by_juniper``, or ``line:<line>`` for a refusal."""
    line = str(line or "").strip() or "unknown"
    if line in CURIOSITY_LINES_FOR_JUNIPER:
        return True, f"line:{line}"
    chat = str(focus_chat_correlation_id or "").strip()
    if str(focus_origin or "").strip() == FOCUS_ORIGIN_JUNIPER and chat:
        return True, "asked_by_juniper"
    return False, f"line:{line}"


def daydream_content_id(caption: str) -> str:
    """Stable identity of a daydream caption, for the tick's 'already said' ledger."""
    text = " ".join(str(caption or "").split())
    if not text:
        return ""
    return "daydream:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def curiosity_finding_key(run_id: str) -> str:
    return f"curiosity_run:{str(run_id or '').strip()}"
