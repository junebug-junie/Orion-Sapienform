"""The memory confirmation loop: Orion asks Juniper about a high-stakes memory, she answers once.

Spec: docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md, sections 3 and 5
(pulled forward from Stage 3, Juniper 2026-10-06). Shadow scope: an answer changes only the
shadow ``episode_memory`` rows and the card she sees; nothing else live reads them yet.

The loop, end to end:

1. **Open** (``open_cards``, orion-memory-consolidation's ticker). A high-stakes memory the
   distiller stored as ``pending_confirmation`` gets one "Orion is asking" card (an ``orion_ask``
   row, ``source_kind='memory_confirmation'``, ``source_ref`` = the loop id) and its
   ``confirmation_loop_id``. At most ``MAX_OPEN_CARDS`` cards are open at once; the rest wait,
   oldest first, until a slot frees. Trace: an ``episode_memory_event`` ``confirm_asked``.
2. **Answer** (Hub, ``scripts/ask_routes.py`` ``POST /api/asks/{id}/resolve``). Confirm, Revise
   (note required) or Reject closes the card AND inserts the one resolution record,
   ``attention_loop_outcome``, in one transaction, then publishes ``AttentionLoopOutcomeV1`` on
   ``orion:attention:loop_outcome``.
3. **Apply** (``apply_outcome``, orion-memory-consolidation). On the bus event, and on a catch-up
   read of the table every tick (``pending_outcomes``), so a lost publish delays an answer but
   never drops it. Idempotent: the apply event's id is derived from the outcome id.
4. **Expire** (``expire_cards``). A card unanswered for ``ASK_TTL`` is closed as ``expired`` and
   the memory becomes ``unconfirmed``. Silence is never a yes, and no outcome row is written.

Card wording is deterministic (``render_question``): the memory's own statement, quoted, framed by
where it came from (voice + channel) and why Orion is asking (``stakes_reason``). Source
monitoring: a memory from an internal channel (reverie, dream, ...) is never framed as something
Juniper said, whatever its voice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --- contract constants -----------------------------------------------------------------------

SOURCE_KIND = "memory_confirmation"
# Kinds that share the 5-card cap and can only be closed through the resolve bridge. open_question
# has no producer yet (spec section 8); it is listed so the cap and the Hub guard already cover it.
RESOLVABLE_KINDS: tuple[str, ...] = (SOURCE_KIND, "open_question")
LOOP_PREFIX = "memory-confirm-"
MAX_OPEN_CARDS = 5
# New memory cards per local day (review of #2517): when Juniper answers promptly, slots free fast,
# and without this the 5-card cap alone would let the queue refill all day.
DAILY_CAP = 3
DEFAULT_TZ = "America/Denver"
ASK_TTL = timedelta(days=7)
ACTOR = "memory.confirmation"
VIA_PANEL = "orion_is_asking"
MAX_NOTE_CHARS = 500
MAX_STATEMENT_CHARS = 400
# Reinforcement on confirmation (spec section 4): strength += 0.2 (cap 1), half-life doubles.
REINFORCE_STEP = 0.2
HALF_LIFE_CAP_DAYS = 365.0

RESOLUTIONS = ("confirmed", "revised", "rejected")
# Answer -> (attention verdict, orion_ask status). Rejected is a dismissal, not a resolution.
RESOLUTION_VERDICT = {"confirmed": "resolved", "revised": "resolved", "rejected": "dismissed"}
RESOLUTION_ASK_STATUS = {"confirmed": "answered", "revised": "answered", "rejected": "dismissed"}

# A revise note must be real wording, not a meta-note ("no", "wrong"). Same floor the distiller
# validator applies to every statement (validate.MIN_STATEMENT_WORDS). Rejection has its own button.
MIN_REVISION_WORDS = 6

CONFIRMATION_NAMESPACE = uuid.UUID("9b0e4c1a-5f7d-4e62-8a3b-c2d1f0e9a8b7")
# pg_advisory_xact_lock key: one card opener at a time, so two replicas cannot both fill slot 5.
OPEN_LOCK_KEY = 0x6D656D636F6E66  # "memconf"


def loop_id_for(memory_id: str) -> str:
    return f"{LOOP_PREFIX}{memory_id}"


def memory_id_from_loop(loop_id: str) -> Optional[str]:
    if not isinstance(loop_id, str) or not loop_id.startswith(LOOP_PREFIX):
        return None
    raw = loop_id[len(LOOP_PREFIX):]
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return None


def ask_id_for(loop_id: str) -> str:
    """uuid5(loop_id): the card insert is idempotent (spec section 5)."""
    return str(uuid.uuid5(CONFIRMATION_NAMESPACE, loop_id))


def outcome_id_for(ask_id: str) -> str:
    """uuid5(ask_id): one panel answer per card, whatever the retries."""
    return str(uuid.uuid5(CONFIRMATION_NAMESPACE, f"outcome|{ask_id}"))


def normalize_statement(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def revision_problem(note: str, current_statement: Optional[str]) -> Optional[str]:
    """Why a revise note cannot become the memory's new wording, or None when it can. Structural
    checks only (no word lists): it must say something (>= MIN_REVISION_WORDS words) and differ
    from the wording it replaces. A note that only rejects belongs on the Reject button."""
    words = str(note or "").split()
    if not words:
        return "revised_needs_note"
    if len(words) < MIN_REVISION_WORDS:
        return "revised_too_short"
    if current_statement is not None and normalize_statement(note) == normalize_statement(current_statement):
        return "revised_unchanged"
    return None


def _event_id(key: str) -> str:
    return str(uuid.uuid5(CONFIRMATION_NAMESPACE, f"event|{key}"))


# --- card wording -----------------------------------------------------------------------------

INTERNAL_CHANNELS = frozenset({"reverie", "curiosity", "dream", "journal", "topic_model"})
_CHANNEL_LABEL = {
    "reverie": "reverie",
    "curiosity": "curiosity runs",
    "dream": "dreams",
    "journal": "journal",
    "topic_model": "topic model",
}

# Why Orion is asking, per stakes category (Juniper's rubric, 2026-10-06). Each category's
# consumer is this line on the card; "none" (low stakes) is never asked at all.
WHY_BY_REASON: dict[str, str] = {
    "health": "It's about health, so I'd rather check than assume.",
    "family_relationships": "It's about your family and the people close to you, so I want to get it right.",
    "juniper_feelings": "It's about how you were feeling, and I don't want to put words in your mouth.",
    "identity_conclusion_about_juniper": (
        "It's my read on who you are, not something you said in so many words."
    ),
    "orion_machinery": "It's a conclusion about how I work, and you can check it better than I can.",
    "orion_asks_direction": "I need your direction on this one.",
    "orion_relationship": "It's about us, so I don't want to decide it alone.",
}
# Labels the validator stores (never a distiller category): why the card is asking anyway.
WHY_BY_VALIDATOR_LABEL: dict[str, str] = {
    "ungrounded_name": "I wrote a name into it that you didn't say, so I want to check it's right.",
}
# The identity line is only true when the memory is Orion's inference. When the memory is Juniper's
# own words (chat, a Juniper voice; the validator keeps that voice only with a verified quote of her
# prompt), "not something you said" would contradict the opener, so the card says why it is heavy.
WHY_IDENTITY_QUOTED = "It's about how you see yourself, and I don't want to keep something that heavy without checking."
# The validator's label for a memory escalated without a category, and pre-v3 rows with none.
WHY_UNJUDGED = "I couldn't tell how personal this is, so I'm checking first."

CLOSER_BY_REASON: dict[str, str] = {
    "identity_conclusion_about_juniper": "Is that fair, and should I keep it?",
    "orion_asks_direction": "Is that the right direction?",
}
CLOSER_DEFAULT = "Want me to remember that?"


def _when(occurred_at: Optional[datetime], created_at: Optional[datetime]) -> str:
    ts = occurred_at or created_at
    if not isinstance(ts, datetime):
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return f" on {ts.strftime('%b')} {ts.day}"


def opener(voice: str, channel: str, when: str = "") -> str:
    """Where the memory came from, in Orion's voice. The channel wins over the voice: anything
    that surfaced on an internal channel is Orion's own, never "you told me"."""
    if channel in INTERNAL_CHANNELS:
        label = _CHANNEL_LABEL.get(channel, channel)
        return f"This came from my own {label}{when}, not from anything you told me. I wrote it down as:"
    if channel == "reading" or voice == "orion_read":
        return f"From something I read{when}, I wrote down:"
    if channel == "graphify" or voice == "orion_self_knowledge":
        return f"From my own code and docs{when}, I wrote down:"
    if channel != "chat":
        return f"I wrote this down{when}:"
    if voice == "juniper_said":
        return f"You told me something{when}, and I wrote it down like this:"
    if voice == "worked_out_together":
        return f"From something we worked out together{when}, I wrote down:"
    return f"From our conversation{when}, this is my own take, which I wrote down as:"


def render_question(
    *,
    statement: str,
    voice: str,
    channel: str,
    stakes_reason: Optional[str],
    occurred_at: Optional[datetime] = None,
    created_at: Optional[datetime] = None,
) -> str:
    text = " ".join(str(statement or "").split())
    if len(text) > MAX_STATEMENT_CHARS:
        text = text[: MAX_STATEMENT_CHARS - 1].rstrip() + "…"
    reason = (stakes_reason or "").strip().lower()
    why = WHY_BY_REASON.get(reason) or WHY_BY_VALIDATOR_LABEL.get(reason, WHY_UNJUDGED)
    closer = CLOSER_BY_REASON.get(reason, CLOSER_DEFAULT)
    if reason == "identity_conclusion_about_juniper" and channel == "chat" and voice in ("juniper_said", "worked_out_together"):
        why, closer = WHY_IDENTITY_QUOTED, CLOSER_DEFAULT
    return f"{opener(voice, channel, _when(occurred_at, created_at))} “{text}” {why} {closer}"


# --- the outcome record (built by the Hub, read by the consumer) ------------------------------


def outcome_features(*, resolution: str, ask_id: str, memory_id: Optional[str], via: str = VIA_PANEL) -> dict:
    return {
        "resolution": resolution,
        "ask_id": ask_id,
        "via": via,
        "memory_id": memory_id,
        "prior_ids": [],
        "related_loop_ids": [],
        "node_ids": [],
    }


@dataclass(frozen=True)
class OutcomeToApply:
    outcome_id: str
    loop_id: str
    verdict: str
    note: str
    features: dict[str, Any]
    actor: str = "juniper"

    @property
    def resolution(self) -> Optional[str]:
        res = str((self.features or {}).get("resolution") or "").strip().lower()
        return res if res in RESOLUTIONS else None

    @property
    def ask_id(self) -> Optional[str]:
        raw = (self.features or {}).get("ask_id")
        return str(raw) if raw else None

    @classmethod
    def from_row(cls, row: Any) -> "OutcomeToApply":
        feats = row["features_at_close"]
        if isinstance(feats, str):
            try:
                feats = json.loads(feats)
            except ValueError:
                feats = {}
        return cls(
            outcome_id=str(row["outcome_id"]),
            loop_id=str(row["loop_id"]),
            verdict=str(row["verdict"]),
            note=str(row["note"] or ""),
            features=dict(feats or {}),
            actor=str(row["actor"] or "juniper"),
        )


# --- SQL (asyncpg) ----------------------------------------------------------------------------

_INSERT_EVENT = """
INSERT INTO episode_memory_event (event_id, memory_id, op, actor, episode_id, outcome_id, evidence, reason, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
ON CONFLICT (event_id) DO NOTHING
RETURNING event_id
"""


async def _event(conn, *, key: str, memory_id: Optional[str], op: str, actor: str, episode_id: Optional[str],
                 outcome_id: Optional[str], evidence: dict, reason: Optional[str], now: datetime) -> bool:
    row = await conn.fetchrow(
        _INSERT_EVENT,
        uuid.UUID(_event_id(key)), uuid.UUID(memory_id) if memory_id else None, op, actor, episode_id,
        outcome_id, json.dumps(evidence, default=str), reason, now,
    )
    return row is not None


def local_day_start(now: datetime, tz_name: str = DEFAULT_TZ) -> datetime:
    from zoneinfo import ZoneInfo

    local = now.astimezone(ZoneInfo(tz_name))
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


# A pending memory whose statement is word-for-word one Juniper already rejected (whitespace and
# case folded) is never asked again: exact match only, no similarity (review of #2517). Matching on
# the referent set + purpose was considered and left as a follow-up: too coarse (one rejected
# "person:sister" memory would silence every later one about her sister).
_SKIP_REJECTED_SQL = """
SELECT m.memory_id::text AS memory_id, m.episode_id, r.memory_id::text AS rejected_memory_id
FROM episode_memory m
JOIN LATERAL (
    SELECT memory_id FROM episode_memory x
    WHERE x.confirmation_state = 'rejected' AND x.memory_id <> m.memory_id
      AND lower(regexp_replace(btrim(x.statement), '\\s+', ' ', 'g'))
        = lower(regexp_replace(btrim(m.statement), '\\s+', ' ', 'g'))
    LIMIT 1
) r ON true
WHERE m.stakes = 'high' AND m.confirmation_state = 'pending_confirmation' AND m.status = 'active'
  AND m.confirmation_loop_id IS NULL
"""


async def open_cards(
    conn,
    *,
    now: Optional[datetime] = None,
    cap: int = MAX_OPEN_CARDS,
    daily_cap: int = DAILY_CAP,
    tz_name: str = DEFAULT_TZ,
) -> list[str]:
    """Open cards for the oldest unasked high-stakes memories, up to the open cap AND the number of
    new cards still allowed today (local day). Returns the loop ids opened. One transaction under an
    advisory lock, so concurrent openers never exceed either cap."""
    now = now or datetime.now(timezone.utc)
    opened: list[str] = []
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1)", OPEN_LOCK_KEY)
        # Do-not-remint: log (once) and hold back exact repeats of a rejected statement.
        held = await conn.fetch(_SKIP_REJECTED_SQL)
        for h in held:
            await _event(conn, key=f"{h['memory_id']}|skipped_rejected_before", memory_id=h["memory_id"],
                         op="confirm_skipped", actor=ACTOR, episode_id=h["episode_id"], outcome_id=None,
                         evidence={"rejected_memory_id": h["rejected_memory_id"]},
                         reason="same_statement_rejected_before", now=now)
        held_ids = [uuid.UUID(h["memory_id"]) for h in held]
        open_n = await conn.fetchval(
            "SELECT count(*) FROM orion_ask WHERE source_kind = ANY($1::text[]) AND status = 'open'",
            list(RESOLVABLE_KINDS),
        )
        today_n = await conn.fetchval(
            "SELECT count(*) FROM orion_ask WHERE source_kind = $1 AND created_at >= $2",
            SOURCE_KIND, local_day_start(now, tz_name),
        )
        slots = max(0, min(int(cap) - int(open_n or 0), int(daily_cap) - int(today_n or 0)))
        if slots == 0:
            return opened
        rows = await conn.fetch(
            """
            SELECT memory_id::text AS memory_id, episode_id, voice, channel, statement, stakes_reason,
                   occurred_at, created_at
            FROM episode_memory
            WHERE stakes = 'high' AND confirmation_state = 'pending_confirmation' AND status = 'active'
              AND confirmation_loop_id IS NULL AND NOT (memory_id = ANY($2::uuid[]))
            ORDER BY created_at, memory_id
            LIMIT $1
            """,
            slots, held_ids,
        )
        for r in rows:
            memory_id = r["memory_id"]
            loop_id = loop_id_for(memory_id)
            # A memory re-asked after the reaper (attempt n > 0) needs a fresh card id: the first
            # card's row still exists, closed.
            attempt = int(await conn.fetchval(
                "SELECT count(*) FROM orion_ask WHERE source_kind = $1 AND source_ref = $2", SOURCE_KIND, loop_id
            ) or 0)
            ask_id = ask_id_for(loop_id if attempt == 0 else f"{loop_id}#{attempt}")
            refs = await conn.fetch(
                "SELECT DISTINCT source_kind, source_id FROM episode_memory_evidence WHERE memory_id = $1 "
                "ORDER BY 1, 2",
                uuid.UUID(memory_id),
            )
            evidence_refs = [f"memory:{memory_id}"] + [f"{e['source_kind']}:{e['source_id']}" for e in refs]
            question = render_question(
                statement=r["statement"], voice=r["voice"], channel=r["channel"],
                stakes_reason=r["stakes_reason"], occurred_at=r["occurred_at"], created_at=r["created_at"],
            )
            await conn.execute(
                """
                INSERT INTO orion_ask (ask_id, asked_of, question, evidence_refs, status, created_at, expires_at,
                                       source_kind, source_ref)
                VALUES ($1, 'juniper', $2, $3::jsonb, 'open', $4, $5, $6, $7)
                ON CONFLICT DO NOTHING
                """,
                ask_id, question, json.dumps(evidence_refs), now, now + ASK_TTL, SOURCE_KIND, loop_id,
            )
            await conn.execute(
                "UPDATE episode_memory SET confirmation_loop_id = $2, updated_at = $3 WHERE memory_id = $1",
                uuid.UUID(memory_id), loop_id, now,
            )
            await _event(conn, key=f"{loop_id}|confirm_asked" + (f"|{attempt}" if attempt else ""),
                         memory_id=memory_id, op="confirm_asked", actor=ACTOR,
                         episode_id=r["episode_id"], outcome_id=None,
                         evidence={"ask_id": ask_id, "loop_id": loop_id, "stakes_reason": r["stakes_reason"],
                                   "attempt": attempt,
                                   "expires_at": (now + ASK_TTL).isoformat()},
                         reason=None, now=now)
            opened.append(loop_id)
    return opened


async def expire_cards(conn, *, now: Optional[datetime] = None) -> list[str]:
    """Close unanswered cards past their 7 days; their memories become ``unconfirmed``.
    Also catches a card some other sweeper already marked expired (orion-sql-writer expires every
    open ask past expires_at). Returns the memory ids moved. No outcome is written: an expiry is
    not an answer."""
    now = now or datetime.now(timezone.utc)
    moved: list[str] = []
    async with conn.transaction():
        await conn.execute(
            "UPDATE orion_ask SET status = 'expired' WHERE source_kind = $1 AND status = 'open' "
            "AND expires_at IS NOT NULL AND expires_at <= $2",
            SOURCE_KIND, now,
        )
        rows = await conn.fetch(
            """
            UPDATE episode_memory m SET confirmation_state = 'unconfirmed', updated_at = $2
            FROM orion_ask a
            WHERE a.source_kind = $1 AND a.status = 'expired' AND a.source_ref = m.confirmation_loop_id
              AND m.confirmation_state = 'pending_confirmation'
            RETURNING m.memory_id::text AS memory_id, m.episode_id, a.ask_id, m.confirmation_loop_id
            """,
            SOURCE_KIND, now,
        )
        for r in rows:
            await _event(conn, key=f"{r['confirmation_loop_id']}|ask_expired", memory_id=r["memory_id"],
                         op="ask_expired", actor=ACTOR, episode_id=r["episode_id"], outcome_id=None,
                         evidence={"ask_id": r["ask_id"]}, reason="no_answer_in_7_days", now=now)
            moved.append(r["memory_id"])
    return moved


PENDING_OUTCOMES_SQL = """
SELECT o.outcome_id, o.loop_id, o.verdict, o.actor, o.note, o.features_at_close
FROM attention_loop_outcome o
WHERE o.loop_id LIKE 'memory-confirm-%'
  AND NOT EXISTS (SELECT 1 FROM episode_memory_event e WHERE e.outcome_id = o.outcome_id)
ORDER BY o.created_at, o.outcome_id
LIMIT $1
"""


async def pending_outcomes(conn, *, limit: int = 50) -> list[OutcomeToApply]:
    """The catch-up read: memory outcomes no ``episode_memory_event`` carries yet."""
    return [OutcomeToApply.from_row(r) for r in await conn.fetch(PENDING_OUTCOMES_SQL, int(limit))]


async def apply_outcome(conn, outcome: OutcomeToApply, *, now: Optional[datetime] = None) -> str:
    """Apply one answer to its memory. Returns what happened (for logs and tests):
    ``confirmed`` | ``revised`` | ``rejected`` | ``already_applied`` | ``already_resolved`` |
    ``orphaned`` | ``invalid`` | ``not_memory``. Every branch except ``not_memory`` writes an
    event carrying the outcome id, so the catch-up read never picks the same outcome twice."""
    now = now or datetime.now(timezone.utc)
    memory_id = memory_id_from_loop(outcome.loop_id)
    if memory_id is None:
        return "not_memory"
    oid = outcome.outcome_id
    base = {"resolution": outcome.resolution, "ask_id": outcome.ask_id, "via": outcome.features.get("via"),
            "verdict": outcome.verdict}
    async with conn.transaction():
        # Lock the memory row FIRST, then check for a prior apply: the bus handler and the catch-up
        # tick can race on one outcome, and the loser must see the winner's committed event.
        mem = await conn.fetchrow(
            """
            SELECT memory_id::text AS memory_id, episode_id, purpose, voice, channel, statement, occurred_at,
                   stakes, stakes_reason, confirmation_state, confirmation_loop_id, strength, half_life_days,
                   reinforcement_count, due_after, expires_at, status
            FROM episode_memory WHERE memory_id = $1 FOR UPDATE
            """,
            uuid.UUID(memory_id),
        )
        done = await conn.fetchval("SELECT 1 FROM episode_memory_event WHERE outcome_id = $1 LIMIT 1", oid)
        if done:
            return "already_applied"

        async def mark(op: str, reason: Optional[str], extra: Optional[dict] = None, mid: Optional[str] = memory_id):
            await _event(conn, key=f"outcome|{oid}|{op}", memory_id=mid, op=op, actor=outcome.actor,
                         episode_id=mem["episode_id"] if mem else None, outcome_id=oid,
                         evidence={**base, **(extra or {})}, reason=reason, now=now)

        if mem is None:
            await mark("outcome_orphaned", "memory_not_found", mid=None)
            return "orphaned"
        if mem["confirmation_state"] not in ("pending_confirmation", "unconfirmed"):
            # First answer wins (spec section 5): a later answer for a settled memory is logged only.
            await mark("outcome_ignored", f"already_{mem['confirmation_state']}")
            return "already_resolved"
        resolution = outcome.resolution
        note = " ".join(outcome.note.split())[:MAX_NOTE_CHARS]
        if resolution is None:
            await mark("outcome_invalid", "unknown_resolution")
            return "invalid"
        if resolution == "revised":
            problem = revision_problem(note, mem["statement"])
            if problem:
                # The Hub refuses these before the card closes; this is the backstop. The memory
                # stays as it was, and the reaper lets it be asked again.
                await mark("outcome_invalid", problem)
                return "invalid"

        half_life = mem["half_life_days"]
        new_half_life = None if half_life is None else min(float(half_life) * 2.0, HALF_LIFE_CAP_DAYS)
        new_strength = min(1.0, float(mem["strength"]) + REINFORCE_STEP)

        if resolution == "confirmed":
            await conn.execute(
                """
                UPDATE episode_memory SET confirmation_state = 'confirmed', voice = 'worked_out_together',
                    strength = $2, half_life_days = $3, reinforcement_count = reinforcement_count + 1,
                    last_reinforced_at = $4, updated_at = $4
                WHERE memory_id = $1
                """,
                uuid.UUID(memory_id), new_strength, new_half_life, now,
            )
            if note:
                await _add_juniper_evidence(conn, memory_id, "juniper_confirmation", outcome, note)
            await mark("confirmed", None, {"prior_voice": mem["voice"], "note": bool(note)})
            return "confirmed"

        if resolution == "rejected":
            await conn.execute(
                "UPDATE episode_memory SET confirmation_state = 'rejected', status = 'rejected', updated_at = $2 "
                "WHERE memory_id = $1",
                uuid.UUID(memory_id), now,
            )
            await mark("rejected", "rejected_by_juniper", {"note": bool(note)})
            return "rejected"

        # revised: Juniper's note is the new wording. The original stays, superseded, so
        # "what did I believe before she corrected me" remains answerable.
        new_id = str(uuid.uuid5(CONFIRMATION_NAMESPACE, f"revised|{memory_id}|{oid}"))
        await conn.execute(
            """
            INSERT INTO episode_memory (
                memory_id, episode_id, purpose, voice, channel, statement, occurred_at, stakes, stakes_reason,
                confirmation_state, confirmation_loop_id, strength, half_life_days, last_reinforced_at,
                reinforcement_count, due_after, expires_at, status, supersedes_memory_id, model_route,
                prompt_version, run_id, created_at, updated_at)
            VALUES ($1, $2, $3, 'worked_out_together', $4, $5, $6, $7, $8, 'confirmed', $9, $10, $11, $12,
                    $13, $14, $15, 'active', $16, NULL, 'juniper_revision', NULL, $12, $12)
            ON CONFLICT (memory_id) DO NOTHING
            """,
            uuid.UUID(new_id), mem["episode_id"], mem["purpose"], mem["channel"], note, mem["occurred_at"],
            mem["stakes"], mem["stakes_reason"], mem["confirmation_loop_id"], new_strength, new_half_life, now,
            int(mem["reinforcement_count"] or 0) + 1, mem["due_after"], mem["expires_at"], uuid.UUID(memory_id),
        )
        # The old quotes supported the OLD wording, so none are copied (review of #2517): the new
        # memory's only evidence is her note, verified as her own words. Referents are not copied
        # either (a revision can change who it is about). History is supersedes_memory_id alone.
        await _add_juniper_evidence(conn, new_id, "confirmation_revise", outcome, note)
        await conn.execute(
            "UPDATE episode_memory SET confirmation_state = 'corrected', status = 'superseded', updated_at = $2 "
            "WHERE memory_id = $1",
            uuid.UUID(memory_id), now,
        )
        await mark("revised", None, {"new_memory_id": new_id})
        await _event(conn, key=f"outcome|{oid}|created", memory_id=new_id, op="created", actor=outcome.actor,
                     episode_id=mem["episode_id"], outcome_id=oid,
                     evidence={**base, "supersedes_memory_id": memory_id}, reason="revised_by_juniper", now=now)
        return "revised"


async def _add_juniper_evidence(conn, memory_id: str, source_kind: str, outcome: OutcomeToApply, note: str) -> None:
    await conn.execute(
        "INSERT INTO episode_memory_evidence (memory_id, source_kind, source_id, quote, quote_sha256, verified) "
        "VALUES ($1, $2, $3, $4, $5, true) ON CONFLICT DO NOTHING",
        uuid.UUID(memory_id), source_kind, outcome.ask_id or outcome.outcome_id, note,
        hashlib.sha256(note.encode("utf-8")).hexdigest(),
    )


async def reap_stuck(conn, *, now: Optional[datetime] = None) -> list[str]:
    """Free memories stuck behind a card that closed without an answer record.

    A ``pending_confirmation`` memory whose card is answered/dismissed (or gone) while no
    ``attention_loop_outcome`` exists for its loop can never move: the opener skips it (it has a
    loop id) and the applier has nothing to apply. That happens if a card is closed outside the
    resolve bridge (an old Hub, a manual UPDATE) or a revise note was refused after the close.
    Clearing the loop id lets the opener ask again with a fresh card. Expired cards are not stuck:
    they move the memory to ``unconfirmed``. An outcome the applier refused as invalid counts as no
    answer. Returns the memory ids freed."""
    now = now or datetime.now(timezone.utc)
    freed: list[str] = []
    async with conn.transaction():
        rows = await conn.fetch(
            """
            SELECT m.memory_id::text AS memory_id, m.episode_id, m.confirmation_loop_id AS loop_id,
                   (SELECT string_agg(a.status, ',') FROM orion_ask a
                     WHERE a.source_kind = $1 AND a.source_ref = m.confirmation_loop_id) AS card_statuses
            FROM episode_memory m
            WHERE m.confirmation_state = 'pending_confirmation' AND m.confirmation_loop_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM orion_ask a WHERE a.source_kind = $1
                              AND a.source_ref = m.confirmation_loop_id AND a.status IN ('open', 'expired'))
              -- No usable answer: no outcome, or only outcomes the applier refused as invalid.
              AND NOT EXISTS (SELECT 1 FROM attention_loop_outcome o WHERE o.loop_id = m.confirmation_loop_id
                              AND NOT EXISTS (SELECT 1 FROM episode_memory_event e
                                              WHERE e.outcome_id = o.outcome_id AND e.op = 'outcome_invalid'))
            FOR UPDATE OF m
            """,
            SOURCE_KIND,
        )
        for r in rows:
            await conn.execute(
                "UPDATE episode_memory SET confirmation_loop_id = NULL, updated_at = $2 WHERE memory_id = $1",
                uuid.UUID(r["memory_id"]), now,
            )
            await _event(conn, key=f"{r['loop_id']}|reaped|{now.isoformat()}", memory_id=r["memory_id"],
                         op="ask_reaped", actor=ACTOR, episode_id=r["episode_id"], outcome_id=None,
                         evidence={"loop_id": r["loop_id"], "card_statuses": r["card_statuses"]},
                         reason="card_closed_without_outcome", now=now)
            freed.append(r["memory_id"])
    return freed


async def run_tick(
    pool: Any,
    *,
    now: Optional[datetime] = None,
    cap: int = MAX_OPEN_CARDS,
    daily_cap: int = DAILY_CAP,
    tz_name: str = DEFAULT_TZ,
) -> dict[str, int]:
    """One pass, in the order that frees slots before filling them: expire, catch up, reap, open.
    One bad outcome is logged and counted (``apply_failed``) and never stops the rest."""
    summary = {"expired": 0, "applied": 0, "apply_failed": 0, "reaped": 0, "opened": 0}
    async with pool.acquire() as conn:
        summary["expired"] = len(await expire_cards(conn, now=now))
        for outcome in await pending_outcomes(conn):
            try:
                result = await apply_outcome(conn, outcome, now=now)
            except Exception:  # noqa: BLE001 -- a poison outcome is retried next tick, never blocks
                logger.exception("memory_confirmation_apply_failed outcome_id=%s loop_id=%s",
                                 outcome.outcome_id, outcome.loop_id)
                summary["apply_failed"] += 1
                continue
            if result not in ("already_applied", "not_memory"):
                summary["applied"] += 1
        summary["reaped"] = len(await reap_stuck(conn, now=now))
        if summary["reaped"]:
            logger.warning("memory_confirmation_reaped count=%s", summary["reaped"])
        summary["opened"] = len(await open_cards(conn, now=now, cap=cap, daily_cap=daily_cap, tz_name=tz_name))
    return summary
