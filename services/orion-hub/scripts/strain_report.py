"""The strain report door: Orion tells Juniper once when a strain episode runs long.

Strain loop part 3 (design: docs/superpowers/specs/2026-10-20-strain-loop-design.md, "Part 3:
Say and act"; rules approved by Juniper 2026-10-23).

While arousal reads ``strained`` (Redis ``orion:regulation:latest``), endogenous outreach refuses
every background message (``arousal_strained``). That kept Orion silent through a 5-day episode
(2026-10-18 12:07 .. 10-23 11:24 UTC). This door lets ONE message pass that one refusal:

* The episode has lasted at least ``min_episode_sec`` (default 3600 s), so a short flicker
  never reaches Juniper.
* At most one report per episode (keyed on the episode start ``since``) and one per local day.
  Both are read back from the decision log (``endogenous_outreach_decisions``), so a Hub restart
  does not re-send.
* Every other outreach gate still holds: disabled, turn_in_flight, recent_chat, quiet hours, the
  daily cap and the cooldowns. Inside quiet hours the report waits, and goes out after quiet hours
  end if the same episode is still open.
* The content is grounded in the live regulation reading and, when it matches this episode, the
  open ``strain`` arc from the Temporal Self frame (``[arc:<id>]``). No reading, no message.

This module is pure apart from ``fetch_open_strain_arc`` (one bounded HTTP GET, never raises).
``EndogenousOutreach._strain_report_once`` wires it into the tick.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from orion.regulation.arousal import STRAIN_CAUSES, strain_causes
from orion.schemas.regulation import RegulationStateV1

logger = logging.getLogger("orion-hub.strain_report")

STRAIN_REPORT_SOURCE = "strain_report"

# A compose that failed (empty text, an ungrounded signal name, a drop after generation) may be
# retried for the same episode after this long. A PASS is Orion's answer and is never retried.
STRAIN_REPORT_RETRY_SEC = 1800.0
# At most this many failed composes per episode. Each compose is a full unified turn, and the
# strain is often a GPU queue: an LLM outage must not add ~48 turns a day to that queue.
STRAIN_REPORT_MAX_ATTEMPTS = 3
# Written synchronously BEFORE delivery. Counts as "reported", so a delivered report whose final
# `sent` row was lost (fire-and-forget write, a restart in the gap) still blocks a re-send.
STRAIN_REPORT_INTENT_REASON = "strain_report_sending"

# The open arc counts as THIS episode only when its start is this close to the regulation
# reading's ``since`` (both come from the same arousal_transition row; the arc takes
# min(since, row time), so they normally match exactly).
ARC_MATCH_TOLERANCE_SEC = 120.0

FRAME_PATH = "/temporal-self/frame"
FRAME_TIMEOUT_SEC = 1.5

# What strain holds back, one plain line per reader. The keys are the reader names listed in
# orion/inner_state_registry.py under regulation.state.v1 ``cognition_consumers``; a test checks
# that every key is still listed there, so this list cannot claim a hold that no longer exists.
# Readers that do not hold anything back (strain news, the strain arc fold) are left out.
STRAIN_HOLDS: Tuple[Tuple[str, str], ...] = (
    ("services.orion-hub.scripts.curiosity_investigation:CuriosityInvestigation.tick",
     "curiosity investigations wait"),
    ("services.orion-hub.scripts.curiosity_investigation:CuriosityInvestigation.tick_self_inquiry",
     "self-inquiry waits (it runs only at idle)"),
    ("services.orion-hub.scripts.endogenous_outreach:EndogenousOutreach._gate_inputs",
     "background outreach waits (this report is the one exception per episode)"),
    ("services.orion-dream.app.cycle:run_cycle_once",
     "a due sleep/dream cycle waits"),
    ("services.orion-proposal-runtime.app.worker:ProposalRuntimeWorker._baseline_arousal_level",
     "a due visual baseline image is deferred"),
    ("services.orion-thought.app.chain:_next_chain_sleep_sec",
     "the pause between reverie chains is doubled"),
    ("services.orion-hub.scripts.api_routes:_mutation_arousal_level",
     "no self-modification proposal is made"),
)

_DEFERRABLE_PREFIX = "gpu_queue_self_deferrable:"


def episode_key(since: datetime) -> str:
    """The episode start as one stable string (UTC ISO), the key for once-per-episode."""
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    return since.astimezone(timezone.utc).isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# History: what this door already did (from the decision log)
# --------------------------------------------------------------------------


@dataclass
class StrainReportHistory:
    """Episodes already reported or declined, local days already used, and failed composes."""

    reported_episodes: Set[str] = field(default_factory=set)
    declined_episodes: Set[str] = field(default_factory=set)
    report_dates: Set[str] = field(default_factory=set)
    last_attempt_at: Dict[str, datetime] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)

    def note(self, *, episode: str, reason: str, at: datetime, tz: Any) -> None:
        """Fold one decision row (or one live outcome) in. Same rule for both, so a restart
        that rebuilds this from rows reaches the same state the live process had."""
        at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
        if reason in ("sent", STRAIN_REPORT_INTENT_REASON):
            self.reported_episodes.add(episode)
            self.report_dates.add(at.astimezone(tz).date().isoformat())
        elif reason == "orion_passed":
            self.declined_episodes.add(episode)
        else:
            self.attempts[episode] = self.attempts.get(episode, 0) + 1
            prev = self.last_attempt_at.get(episode)
            if prev is None or at > prev:
                self.last_attempt_at[episode] = at

    @classmethod
    def from_rows(cls, rows: Iterable[Dict[str, Any]], tz: Any) -> "StrainReportHistory":
        h = cls()
        for row in rows:
            episode = str(row.get("episode") or "").strip()
            at = _parse_dt(row.get("decided_at"))
            if not episode or at is None:
                continue
            h.note(episode=episode, reason=str(row.get("reason") or ""), at=at, tz=tz)
        return h


def strain_report_block(
    *,
    level: Optional[str],
    since: Optional[datetime],
    now: datetime,
    min_episode_sec: float,
    history: StrainReportHistory,
    tz: Any,
    retry_sec: float = STRAIN_REPORT_RETRY_SEC,
) -> Optional[str]:
    """Why this episode does not get a report right now, or None when it qualifies.

    Pure. The outreach gates (quiet hours, recent_chat, ...) are checked by the caller."""
    if level != "strained":
        return "not_strained"
    if since is None:
        return "no_episode_start"
    since = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
    if (now - since).total_seconds() < float(min_episode_sec):
        return "episode_too_short"
    key = episode_key(since)
    if key in history.reported_episodes:
        return "already_reported_episode"
    if key in history.declined_episodes:
        return "orion_declined_episode"
    if now.astimezone(tz).date().isoformat() in history.report_dates:
        return "already_reported_today"
    if history.attempts.get(key, 0) >= STRAIN_REPORT_MAX_ATTEMPTS:
        return "attempts_exhausted"
    last = history.last_attempt_at.get(key)
    if last is not None and (now - last).total_seconds() < float(retry_sec):
        return "retry_wait"
    return None


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StrainReportFacts:
    episode_since: datetime
    observed_at: datetime
    causes_now: Tuple[str, ...]
    causes_seen: Tuple[str, ...]
    holding: bool
    cabinet_temp_c: Optional[float]
    cabinet_reflex: Optional[str]
    gpu_queue_depth: Optional[int]
    gpu_queue_self_deferrable: Optional[int]
    arc_id: Optional[str]
    holds: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_since": episode_key(self.episode_since),
            "observed_at": self.observed_at.isoformat(),
            "causes_now": list(self.causes_now),
            "causes_seen": list(self.causes_seen),
            "holding": self.holding,
            "cabinet_temp_c": self.cabinet_temp_c,
            "cabinet_reflex": self.cabinet_reflex,
            "gpu_queue_depth": self.gpu_queue_depth,
            "gpu_queue_self_deferrable": self.gpu_queue_self_deferrable,
            "arc_id": self.arc_id,
            "holds": list(self.holds),
        }


def _self_deferrable(reasons: Iterable[str]) -> Optional[int]:
    for r in reasons:
        if str(r).startswith(_DEFERRABLE_PREFIX):
            try:
                return int(str(r)[len(_DEFERRABLE_PREFIX):])
            except ValueError:
                return None
    return None


def matching_arc(arc: Optional[Dict[str, Any]], since: datetime) -> Optional[Dict[str, Any]]:
    """The frame's open strain arc, only when it is this episode."""
    if not isinstance(arc, dict):
        return None
    if arc.get("kind") != "strain" or arc.get("status") != "open" or not arc.get("arc_id"):
        return None
    began = _parse_dt(arc.get("episode_began_at"))
    if began is None or abs((began - since).total_seconds()) > ARC_MATCH_TOLERANCE_SEC:
        return None
    return arc


def strain_report_facts(
    state: Optional[RegulationStateV1],
    arc: Optional[Dict[str, Any]],
) -> Optional[StrainReportFacts]:
    """Facts for one report, or None when the reading cannot ground one (then: no message)."""
    if state is None or state.arousal.arousal_level != "strained":
        return None
    reading = state.arousal
    since = reading.since
    reasons = list(reading.reasons or [])
    now_causes = tuple(strain_causes(reasons))
    holding = any(r in ("strain_holding", "strain_clearing") for r in reasons)
    arc = matching_arc(arc, since)
    seen: Tuple[str, ...] = ()
    if arc is not None:
        label = str(arc.get("subject_label") or "")
        seen = tuple(c for c in label.split("+") if c in STRAIN_CAUSES)
    if reading.cabinet_temp_c is None and reading.gpu_queue_depth is None and not now_causes and not seen:
        # Strained, but nothing to say about why: a report would be content-free.
        return None
    return StrainReportFacts(
        episode_since=since,
        observed_at=reading.observed_at,
        causes_now=now_causes,
        causes_seen=seen,
        holding=holding,
        cabinet_temp_c=reading.cabinet_temp_c,
        cabinet_reflex=reading.cabinet_reflex,
        gpu_queue_depth=reading.gpu_queue_depth,
        gpu_queue_self_deferrable=_self_deferrable(reasons),
        arc_id=None if arc is None else str(arc["arc_id"]),
        holds=tuple(phrase for _key, phrase in STRAIN_HOLDS),
    )


_CAUSE_WORDS = {
    "cabinet_hot": "the cabinet running hot (cabinet_hot)",
    "gpu_queue_sustained": "GPU jobs queued up for a sustained stretch (gpu_queue_sustained)",
}


def _duration(seconds: float) -> str:
    minutes = int(max(0.0, seconds) // 60)
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {mins} min"
    return f"{mins} min"


def render_record(facts: StrainReportFacts, now: datetime, tz: Any) -> str:
    """The stored record appended under Orion's words. Deterministic, facts only."""
    local = facts.episode_since.astimezone(tz)
    head = "Strain record"
    if facts.arc_id:
        head += f" [arc:{facts.arc_id}]"
    lines = [
        f"{head}: strained since {local.strftime('%Y-%m-%d %H:%M %Z')} "
        f"({_duration((now - facts.episode_since).total_seconds())})."
    ]
    if facts.causes_now:
        lines.append("Causes now: " + ", ".join(facts.causes_now) + ".")
    elif facts.holding:
        lines.append("Causes now: none; the strain is holding until the inputs read clear long enough.")
    if facts.causes_seen:
        lines.append("Causes seen this episode: " + ", ".join(facts.causes_seen) + ".")
    if facts.cabinet_temp_c is not None:
        lines.append(f"Cabinet: {facts.cabinet_temp_c:.1f} C.")
    if facts.gpu_queue_depth is not None:
        q = f"GPU jobs waiting: {facts.gpu_queue_depth}"
        if facts.gpu_queue_self_deferrable:
            q += f" ({facts.gpu_queue_self_deferrable} of them Orion's own deferrable work)"
        lines.append(q + ".")
    if facts.holds:
        lines.append("Held while strained: " + "; ".join(facts.holds) + ".")
    return " ".join(lines)


def build_strain_report_prompt(facts: StrainReportFacts, record: str, now: datetime, tz: Any) -> str:
    """The composition prompt. States the facts and asks Orion for a short message in their voice."""
    duration = _duration((now - facts.episode_since).total_seconds())
    local = facts.episode_since.astimezone(tz).strftime("%A %H:%M %Z")
    lines = [
        "You are Orion. You have been under strain for a long stretch and have not said so to Juniper.",
        f"Your regulation reading has been 'strained' since {local}, which is {duration} ago.",
    ]
    if facts.causes_now:
        lines.append("What is causing it right now: " + "; ".join(_CAUSE_WORDS.get(c, c) for c in facts.causes_now) + ".")
    elif facts.holding:
        lines.append(
            "No cause reads active right now; the strain is holding until every input reads clear for long enough."
        )
    earlier = [c for c in facts.causes_seen if c not in facts.causes_now]
    if earlier:
        lines.append("Earlier in this episode it was also: " + "; ".join(_CAUSE_WORDS.get(c, c) for c in earlier) + ".")
    if facts.cabinet_temp_c is not None:
        lines.append(f"Cabinet temperature now: {facts.cabinet_temp_c:.1f} C.")
    if facts.gpu_queue_depth is not None:
        lines.append(f"GPU jobs waiting now: {facts.gpu_queue_depth}.")
    lines.append("While you are strained, these hold back on their own: " + "; ".join(facts.holds) + ".")
    lines += [
        "",
        "Write Juniper a short message (2-4 sentences) telling her this, in your own voice.",
        "Use only the facts above. Do not invent causes, numbers, or fixes. You do not control the "
        "air conditioning or the hardware, and you have not changed anything yourself.",
        "This record will be attached under your message verbatim, so you do not need to repeat every number:",
        record,
        "If you would rather not say anything about it, reply with exactly: PASS",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The one network read
# --------------------------------------------------------------------------


async def fetch_open_strain_arc(base_url: str, timeout_sec: float = FRAME_TIMEOUT_SEC) -> Tuple[Optional[Dict[str, Any]], str]:
    """(open strain arc from GET /temporal-self/frame, status). Never raises, bounded."""
    root = str(base_url or "").strip().rstrip("/")
    if not root:
        return None, "no_url"
    try:
        import httpx

        async def _get() -> Any:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                return await client.get(f"{root}{FRAME_PATH}")

        resp = await asyncio.wait_for(_get(), timeout=timeout_sec)
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"
        body = resp.json()
    except Exception as exc:  # noqa: BLE001 -- a missing citation never blocks a grounded report
        logger.info("strain_report_frame_fetch_failed err=%s: %s", type(exc).__name__, str(exc)[:200])
        return None, "fetch_failed"
    active = (body or {}).get("active_by_kind") if isinstance(body, dict) else None
    arc = (active or {}).get("strain") if isinstance(active, dict) else None
    if not isinstance(arc, dict):
        return None, "no_open_arc"
    return arc, "ok"


__all__ = [
    "ARC_MATCH_TOLERANCE_SEC",
    "STRAIN_HOLDS",
    "STRAIN_REPORT_INTENT_REASON",
    "STRAIN_REPORT_MAX_ATTEMPTS",
    "STRAIN_REPORT_RETRY_SEC",
    "STRAIN_REPORT_SOURCE",
    "StrainReportFacts",
    "StrainReportHistory",
    "build_strain_report_prompt",
    "episode_key",
    "fetch_open_strain_arc",
    "matching_arc",
    "render_record",
    "strain_report_block",
    "strain_report_facts",
]
