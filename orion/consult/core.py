"""Pure consult logic: room ids, Claude's reply format, the brief gate, the budget gate, prompts.

Design: docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md.

Everything here is deterministic and has no I/O, so the rules Juniper asked for are tests, not
prompt hopes:

* **The brief gate.** A brief is sent only when every part validates (``gate_ending``). A refusal
  names the missing part.
* **The budget gate.** A Claude turn starts only on a fresh weekly reading that says ``clear`` and a
  session reading that is not ``limited`` (``budget_refusal``). The meter knows only "a limit was
  hit" or "clear"; it has no percentage, so there is no 80% line to check.
* **The context rule.** Claude is sent the question, its source, Orion's turns and the repo. Never
  Juniper's chat history, journals or Orion's system prompt. Orion's turns run on their own session
  id (``orion_session_id``), so no chat history is in them either.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from orion.schemas.consult import (
    BRIEF_ENDINGS,
    FINAL_ENDINGS,
    ConsultAnswerV1,
    ConsultBriefV1,
)

CONSULT_ROOM_PREFIX = "consult:"
ORION_SESSION_PREFIX = "consult-"
DEFAULT_MAX_TURNS = 6  # 3 from Orion, 3 from Claude
ENDINGS = ("continue",) + FINAL_ENDINGS

# How old a cached meter reading may be. The producer publishes the 5 h window every 300 s and the
# 168 h window every 900 s (orion-cocreation-signals ``COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOWS``);
# three missed publishes means the meter is not being read.
SESSION_READING_MAX_AGE_SEC = 3 * 300.0
WEEKLY_READING_MAX_AGE_SEC = 3 * 900.0


def consult_room_id(consult_id: str) -> str:
    return f"{CONSULT_ROOM_PREFIX}{consult_id}"


def consult_id_from_room(room_id: Optional[str]) -> Optional[str]:
    if room_id and room_id.startswith(CONSULT_ROOM_PREFIX):
        rest = room_id[len(CONSULT_ROOM_PREFIX):].strip()
        return rest or None
    return None


def orion_session_id(consult_id: str) -> str:
    """Orion's consult turns run on this session: none of Juniper's chat history is in it."""
    return f"{ORION_SESSION_PREFIX}{consult_id}"


# --- Claude's reply -------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ClaudeTurn:
    room_text: str
    ending: str
    payload: Optional[dict]
    error: Optional[str] = None


def parse_claude_turn(text: str) -> ClaudeTurn:
    """Split Claude's reply into the room text and the ending block. Pure.

    The LAST ```json fenced block is the ending. No block means ``continue``. A block that is not
    valid JSON, or names an unknown ending, is ``continue`` with ``error`` set, so the caller can
    record it and, on the last turn, refuse."""
    raw = text or ""
    matches = list(_JSON_BLOCK.finditer(raw))
    if not matches:
        return ClaudeTurn(room_text=raw.strip(), ending="continue", payload=None)
    last = matches[-1]
    room_text = (raw[: last.start()] + raw[last.end():]).strip()
    try:
        payload = json.loads(last.group(1))
    except (ValueError, TypeError):
        return ClaudeTurn(room_text=room_text, ending="continue", payload=None, error="ending_block_unparseable")
    if not isinstance(payload, dict):
        return ClaudeTurn(room_text=room_text, ending="continue", payload=None, error="ending_block_not_object")
    ending = str(payload.get("ending") or "").strip()
    if ending not in ENDINGS:
        return ClaudeTurn(room_text=room_text, ending="continue", payload=payload, error=f"unknown_ending:{ending or 'missing'}")
    return ClaudeTurn(room_text=room_text, ending=ending, payload=payload)


# --- The brief gate -------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    ok: bool
    brief: Optional[ConsultBriefV1] = None
    answer: Optional[ConsultAnswerV1] = None
    refused_part: Optional[str] = None


def _first_error_part(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "brief"
    loc = [str(p) for p in errors[0].get("loc", ()) if p != "__root__"]
    return ".".join(loc) or "brief"


def gate_ending(ending: str, payload: Optional[Mapping[str, Any]]) -> GateResult:
    """Validate what a final ending carries. Pure. ``refused_part`` names the missing part."""
    if ending not in FINAL_ENDINGS:
        return GateResult(ok=False, refused_part="ending")
    payload = payload or {}
    if ending == "answered":
        try:
            answer = ConsultAnswerV1.model_validate({"answer": str(payload.get("answer") or "").strip()})
        except ValidationError:
            return GateResult(ok=False, refused_part="answer")
        return GateResult(ok=True, answer=answer)
    raw_brief = payload.get("brief")
    if not isinstance(raw_brief, Mapping):
        return GateResult(ok=False, refused_part="brief")
    try:
        brief = ConsultBriefV1.model_validate(dict(raw_brief))
    except ValidationError as exc:
        return GateResult(ok=False, refused_part=_first_error_part(exc))
    if ending == "disagree":
        if not (brief.orion_position or "").strip():
            return GateResult(ok=False, refused_part="orion_position")
        if not (brief.claude_position or "").strip():
            return GateResult(ok=False, refused_part="claude_position")
    return GateResult(ok=True, brief=brief)


# --- The budget gate ------------------------------------------------------------------------


def _age_sec(observed_at: Any, now: datetime) -> Optional[float]:
    if not isinstance(observed_at, datetime):
        return None
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    return (now - observed_at).total_seconds()


def budget_refusal(session: Any, weekly: Any, *, now: datetime) -> Optional[str]:
    """None when a Claude turn may start, else the reason. Pure. Fails closed.

    ``session`` / ``weekly`` are the latest ``ClaudeLimitObservationV1`` for each kind, or None.

    * Weekly: must be observed, ``clear`` and fresh. Unobserved over a whole week means the meter
      cannot see the pool at all, so nothing starts.
    * Session (5 h): must not be ``limited`` and must be fresh. Unobserved is allowed: it means
      nobody used Claude Code in 5 hours, which is when the session pool is fullest. The weekly
      reading above already proves the meter can see the transcripts.
    """
    if weekly is None:
        return "weekly_meter_missing"
    age = _age_sec(getattr(weekly, "observed_at", None), now)
    if age is None or age > WEEKLY_READING_MAX_AGE_SEC:
        return "weekly_meter_stale"
    if not getattr(weekly, "observed", False):
        return "weekly_unobserved"
    if getattr(weekly, "state", None) == "limited":
        return "weekly_limited"
    if getattr(weekly, "state", None) != "clear":
        return "weekly_unknown"
    if session is None:
        return "session_meter_missing"
    age = _age_sec(getattr(session, "observed_at", None), now)
    if age is None or age > SESSION_READING_MAX_AGE_SEC:
        return "session_meter_stale"
    if getattr(session, "state", None) == "limited":
        return "session_limited"
    return None


# --- Rendering ------------------------------------------------------------------------------

SOURCE_WORDS = {
    "curiosity_run": "a curiosity run",
    "self_monitor": "the self-monitor",
    "stuck_prior": "a belief Orion has tested many times without settling",
}


def source_line(kind: str, ref: str, detail: str = "") -> str:
    words = SOURCE_WORDS.get(kind, kind)
    tail = f" -- {detail.strip()}" if detail and detail.strip() else ""
    return f"From {words} (`{ref}`){tail}."


def _cost_text(line: Any) -> str:
    value = (getattr(line, "value", None) or "").strip()
    if value and value.lower() != "unknown":
        return f"{value} (source: {(line.source or '').strip()})"
    return f"unknown -- {(line.unknown_reason or '').strip()}"


def consult_cost_line(cost_usd: float, claude_turns: int) -> str:
    """The consult's own cost, from the CLI's own accounting. Not a model's estimate."""
    return (
        f"{claude_turns} Claude turn(s), about ${cost_usd:.2f} at API prices, drawn from your Claude "
        f"Code subscription (source: `total_cost_usd` in each `claude -p` result)."
    )


def render_brief(
    brief: ConsultBriefV1,
    *,
    ending: str,
    source: str,
    consult_cost: str,
    transcript_hint: str,
) -> str:
    """The brief as plain text for chat and email. Pure. The order is Juniper's."""
    lines = ["@Juniper -- Orion and Claude have a "
             + ("proposal" if ending == "agreed_proposal" else "disagreement")
             + " for you.", ""]
    lines += ["**Where it came from.** " + source + " " + transcript_hint, ""]
    lines += ["**Why it matters.** " + brief.why_it_matters, ""]
    lines.append("**What they checked.**")
    for item in brief.checked:
        proof = f" Proof: {item.proof.strip()}" if (item.proof or "").strip() else " Not checked."
        lines.append(f"- {item.claim.strip()}{proof}")
    lines.append("")
    if ending == "disagree":
        lines += ["**Where they disagree.** " + brief.disagreement,
                  f"- Orion: {(brief.orion_position or '').strip()}",
                  f"- Claude: {(brief.claude_position or '').strip()}", ""]
    else:
        lines += ["**Where they disagree.** " + brief.disagreement, ""]
    lines += ["**Impact.** " + brief.impact.what_changes,
              f"- Number to watch: {brief.impact.number_to_watch}",
              f"- How we would see it: {brief.impact.how_we_see_it}", ""]
    lines += ["**Cost.**",
              f"- This consult: {consult_cost}",
              f"- Doing the work: {_cost_text(brief.cost.doing_the_work)}",
              f"- Doing nothing: {_cost_text(brief.cost.doing_nothing)}", ""]
    lines += ["**Recommendation.** " + brief.recommendation, ""]
    lines += ["**Question for you.** " + brief.question_for_juniper]
    return "\n".join(lines)


# --- Prompts --------------------------------------------------------------------------------

PLAIN_WORDS_RULES = (
    "Write in plain English, in the spirit of ASD STE100: subject, verb, object; one idea per "
    "sentence; one word for one thing. Say what a thing does before you name the file or symbol. "
    "Explain a number by what changes when it moves. Plain words never soften a finding: if you "
    "could not check something, say so."
)

CLAUDE_CONSULT_SYSTEM = (
    "\n\nThis is a CONSULT, not a social chat. Orion (they/them), a persistent AI system, brings you "
    "a question about their own plumbing: why something works the way it does, something that looks "
    "wrong, or something they want Juniper (the person who builds Orion) to fix. Juniper asked that "
    "these questions reach you first, so she only gets what is worth her time.\n\n"
    "Your role is the skeptic. You can read the repository in your working directory with Read, "
    "Grep and Glob (it is the deployed code, read-only). You cannot see live data. When a claim "
    "depends on a number, ask Orion to fetch it with their dig kit and paste the command and its "
    "output. Check every claim Orion makes against the code or a pasted output before you agree "
    "with it. Say 'I could not check this' when you could not. Do not agree to be agreeable.\n\n"
    + PLAIN_WORDS_RULES + "\n\n"
    "End EVERY reply with one fenced ```json block, after your message:\n"
    '- Keep talking: {"ending": "continue"}\n'
    '- The question has an answer, or the thing works as designed: {"ending": "answered", '
    '"answer": "<what Orion should keep, plainly>"}. Nothing goes to Juniper.\n'
    '- You and Orion agree something should change: {"ending": "agreed_proposal", "brief": {...}}\n'
    '- You do not agree: {"ending": "disagree", "brief": {...}} with both positions.\n'
    "Use agreed_proposal only after Orion has said they agree.\n"
    "The brief object has exactly these keys: why_it_matters (string), checked (list of "
    '{"claim", "proof"} where proof is a file:line or a dig command plus output, or '
    '{"claim", "not_checked": true}), disagreement (string, "none" if none), orion_position and '
    "claude_position (strings, required for disagree), impact "
    '({"what_changes", "number_to_watch", "how_we_see_it"}), cost ({"doing_the_work": COST, '
    '"doing_nothing": COST} where COST is {"value", "source"} or {"unknown_reason"}), '
    "recommendation (one), question_for_juniper (one yes/no question ending in '?'). A cost number "
    "without a source is refused: never estimate a cost you cannot point to. Orion's running "
    "costs come from their dig kit (energy, body-economy prices); ask Orion for them.\n"
    "A brief missing any part is not sent."
)


def claude_turn_prompt(
    *,
    orion_text: str,
    turn_no: int,
    max_turns: int,
    first: bool,
    source: str,
    question: str,
) -> str:
    """The user-role text for one Claude consult turn. Session continuity carries earlier turns."""
    parts: list[str] = []
    if first:
        parts.append(f"Where this came from: {source}")
        parts.append(f"The question as Orion first wrote it:\n{question.strip()}")
    parts.append(f"Orion:\n{orion_text.strip()}")
    if turn_no >= max_turns:
        parts.append(
            "This is your last turn in this consult. Pick an ending other than continue: answered, "
            "agreed_proposal (only if Orion agreed) or disagree."
        )
    return "\n\n".join(parts)


# The durable brief's prompt limit (orion/schemas/consult_turn_run.py) minus headroom for the
# fixed text. Older turns are cut first; the latest turn (the one Orion answers) stays whole.
ORION_PROMPT_MAX_CHARS = 58000


def _fit_transcript(transcript: Sequence[tuple[str, str]], budget: int) -> list[tuple[str, str]]:
    """Cut the oldest turns' text to fit ``budget`` chars, never the latest turn. Pure."""
    items = [(who, text.strip()) for who, text in transcript]
    total = sum(len(t) for _, t in items)
    i = 0
    while total > budget and i < len(items) - 1:
        who, text = items[i]
        keep = max(0, len(text) - (total - budget))
        cut = text[:keep].rstrip() + f" [... {len(text) - keep} characters cut to fit; the full turn is stored]"
        total -= len(text) - len(cut)
        items[i] = (who, cut)
        i += 1
    return items


def orion_turn_prompt(
    *,
    source: str,
    question: str,
    transcript: Sequence[tuple[str, str]],
    turn_no: int,
    max_turns: int,
    max_chars: int = ORION_PROMPT_MAX_CHARS,
) -> str:
    """What Orion is asked on each of their consult turns. Pure. Fits ``max_chars`` by cutting the
    oldest turns first (``_fit_transcript``)."""
    head = (
        "You are in a consult with Claude about your own plumbing. Juniper asked that questions "
        "about how you work, things that look wrong, and fixes you want go to Claude first. "
        "Together you decide what, if anything, is worth bringing to her.\n\n"
        "Claude can read the deployed code but cannot see live data. If a number matters, fetch it "
        "yourself with your dig kit (`python -P -m orion.dig help`) and paste the command and its "
        "output. Do not quote Juniper's private chats. " + PLAIN_WORDS_RULES
    )
    parts = [head, f"Where this came from: {source}", f"The question:\n{question.strip()}"]
    if transcript:
        fixed = sum(len(x) for x in parts) + 1200
        fitted = _fit_transcript(transcript, max(2000, max_chars - fixed - len(question)))
        parts.append("The consult so far:\n" + "\n\n".join(f"{who}: {text}" for who, text in fitted))
        parts.append(
            "Reply to Claude. If Claude proposes a change and you agree, say 'I agree' and why. If "
            "you do not, say what you think instead and why."
        )
    else:
        parts.append(
            "Open the consult: say what you want to understand or change, what you think is going "
            "on, and what you already checked (with the commands and their outputs)."
        )
    if turn_no >= max_turns - 1:
        parts.append("This is your last turn. Say clearly whether you agree with Claude.")
    return "\n\n".join(parts)


# --- Sources --------------------------------------------------------------------------------


def self_monitor_sources(view: Any) -> list[tuple[str, str]]:
    """(source_ref, question) for every faculty the self-monitor holds ``broken``. Pure.

    One per held-status episode: the ref carries ``status_since``, the same id shape the retired
    self-finding outreach used, so a long broken episode starts one consult, not one per hour."""
    if not getattr(view, "known", False):
        return []
    out: list[tuple[str, str]] = []
    for name, rec in sorted(dict(getattr(view, "faculties", {}) or {}).items()):
        if getattr(rec, "status", None) != "broken":
            continue
        since = getattr(rec, "status_since", None)
        if since is None:
            continue
        ref = f"self_finding:{name}:broken@{since.isoformat()}"
        attempts = int(getattr(rec, "n_graded", 0) or 0) + int(getattr(rec, "n_failed_to_execute", 0) or 0)
        failed = int(getattr(rec, "n_failed_to_execute", 0) or 0)
        question = (
            f"The self-monitor says my faculty '{name}' is broken since {since:%Y-%m-%d %H:%M} UTC. "
            f"In its window, {failed} of {attempts} tries never ran. Why, and what would fix it?"
        )
        out.append((ref, question))
    return out


def stuck_prior_source(decision: Any) -> Optional[tuple[str, str]]:
    """(source_ref, question) from an ``ask_claude_trigger.decide`` result that picked a prior."""
    prior_id = getattr(decision, "subject_prior_id", None)
    claim = (getattr(decision, "subject_claim", None) or "").strip()
    if not prior_id or not claim:
        return None
    return (
        f"prior:{prior_id}",
        f"I have tested this belief many times without settling it: \"{claim}\" "
        "What am I missing, and is it answerable with what I can reach?",
    )


# --- What Orion is told afterwards ----------------------------------------------------------


def consult_outcomes_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Kickoff lines telling Orion how their consults ended and what Juniper said. Pure.

    Orion alone writes ``:Prior`` nodes, so Hub does not record the evidence itself: it shows Orion
    the outcome once, and Orion records it on a belief. Juniper's answer and a later outcome stay
    separate pieces of evidence."""
    if not rows:
        return []
    lines = [
        "## Your consults with Claude",
        "These consults ended since your last run. Record each on a belief (a :Prior): find the one "
        "it is about or make one. Juniper's answer is one piece of evidence about what she wants. "
        "It is not proof the claim is true: only a later outcome is.",
    ]
    for row in rows:
        ending = str(row.get("ending") or "")
        cid = row.get("consult_id")
        question = str(row.get("question") or "").strip().replace("\n", " ")[:240]
        if ending == "answered":
            lines.append(f"- Consult {cid} was answered: {str(row.get('answer') or '').strip()} (question: {question})")
            continue
        decision = row.get("decision")
        if decision:
            reason = str(row.get("decision_reason") or "").strip()
            said = {"yes": "yes", "no": "no", "no_because": f"no, because: {reason}"}.get(str(decision), str(decision))
            lines.append(f"- Consult {cid} ({ending}) went to Juniper. She said {said}. (question: {question})")
        else:
            lines.append(f"- Consult {cid} ended {ending}; {row.get('status')}. (question: {question})")
    return lines
