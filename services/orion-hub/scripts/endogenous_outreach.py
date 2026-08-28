"""Endogenous outreach — Orion opens a conversation Juniper did not start.

REAL TRIGGER (2026-08-16, replacing the coin-flip stub this module ran on
since 2026-08-14). ``_should_roll()`` now asks
``scripts.tension_outreach_trigger.current_run()``: has the SAME target
(node_id) been winning `orion.attention.tension`'s live Borda competition for
a sustained, unbroken run of real ticks, right now -- not a single blip. See
that module's own docstring for the full account, including why the bar
(``MIN_RUN_LENGTH``) is derived from a real replay of live history rather than
guessed. The message itself is generated from live substrate signals and real
chat history, and lands on the same three rails a normal turn uses -- none of
that changed.

LEVEL-AWARE, NOT JUST CHANGE-AWARE (2026-08-19). The trigger's own reason
object now also carries ``sustained_load_pressure`` -- a real, currently-
loaded reading (`orion.field.significance`, `loaded_steady` regime, no
adaptive baseline), not a second change-detector. `build_outreach_prompt`
below states it as a separate real fact alongside the deviation-run fact
when it is nonzero; it does not put words like "worried" in Orion's mouth --
that judgment is left to generation, grounded in the real numbers. See
`scripts.tension_outreach_trigger`'s module docstring for the full account
of what this can and cannot honestly claim.

THROUGH THE REAL UNIFIED TURN, NOT A LOOKALIKE (2026-08-19). Generation used
to call ``CortexGatewayClient.chat()`` directly -- a bare bus RPC to
``orion-cortex-gateway`` that never touched ``orion-harness-governor`` at
all: no fcc motor, no substrate appraisal/reflect/voice finalize beats, no
post-turn learning closure, no audit artifact, and (root-caused live,
2026-08-19) a verb-less default (``chat_general``) whose hidden
``synthesize_chat_stance_brief`` pre-step sent a 15.7k-char internal system
prompt while requesting 8000 completion tokens on the ``quick`` route with
no larger fallback lane configured -- overflowing context on every single
attempt. Juniper's call once this was traced: outreach should go through
``orion.hub.turn_orchestrator.execute_unified_turn`` -- the SAME function
``websocket_handler.py`` calls for a real ``client_mode == "orion"`` turn --
not a cheaper substitute, "if orion is going to reach out, it needs to be
real."

This is a bigger change than swapping which client generates text.
``execute_unified_turn`` starts with a real ``emit_observation()`` call (the
outreach prompt is recorded into Orion's own observation stream, same as
anything Juniper says) and a real ``ThoughtClient.react()`` stance
evaluation that can ``defer``/``refuse`` the turn -- there is no shallower
entry point: ``HarnessRunRequestV1`` requires a ``thought_event``, so
reaching the harness governor at all means going through Thought first.
Accepted deliberately: Thought's own defer/refuse is now the honest
"something else is happening, don't interrupt" signal for an unsolicited
turn, not just an infrastructure-availability check.

``payload={"no_write": True, ...}`` suppresses ONLY ``execute_unified_turn``'s
own chat-history persistence (``_publish_unified_turn_chat_history``) --
this module's own three delivery rails below (unchanged) remain the sole
persistence path, so an outreach turn keeps its ``endogenous_outreach`` tag
and ``unsolicited`` client_meta instead of landing as an untagged duplicate
row next to whatever real persistence the governor would otherwise do.
Permissions are NOT payload-driven here: ``execute_unified_turn`` hard-codes
every unified turn's ``ContextExecPermissionV1`` to read-only (write_*/
mutate_runtime/network_enabled/shell_enabled all default ``False``) --
stricter than the old direct-call path's own
``tool_execution_policy``/``action_execution_policy`` options ever were, so
nothing new to pin here.

Not carried over: ``continuity_messages`` (the per-connection UI history
buffer real turns build from ``websocket_handler.py``'s own ``history``
list) -- outreach has no live browser turn to build one from.
``build_outreach_prompt`` already folds real recent turns into the prompt
TEXT itself via ``_fetch_recent_turns`` (unchanged), so this is a narrower,
not an absent, substitute.

``llm_route``/``HUB_ENDOGENOUS_OUTREACH_LLM_ROUTE`` are gone (killed, not
deprecated) -- route selection is no longer Hub's decision to make; it is
whatever the harness governor picks for a real turn, identically.

Delivery (all three already existed before this module; none are new rails):

  1. In-process fan-out to every connected Hub websocket's outbound queue
     (``websocket_handler.drain_queue``) as ``{"kind": "orion_outreach"}``,
     which the frontend renders as a normal Orion chat bubble.
  2. ``chat.history.message.v1`` on the bus with ``role="assistant"``
     (``chat_history.publish_chat_history``), persisted by orion-sql-writer.
     NOTE: this makes the outreach durable *as data* only. Hub's frontend has
     no conversation-restore fetch at all, so a reload does not bring the
     bubble back -- rail 3 is what a returning browser actually sees. Giving
     the UI a real restore path is a separate piece of work.
  3. ``HubNotificationEvent`` on ``NOTIFY_IN_APP_CHANNEL`` so the outreach is
     visible even with no browser open — same pattern as
     ``bus_synaptic_trigger_notifier``.

Single-process assumption: rail 1 is in-process. Hub runs one uvicorn worker
(``Dockerfile`` CMD has no ``--workers``); if that ever changes, rail 1 must
move onto the bus like rails 2 and 3 already are.

Safety posture — this must never disturb a real turn:

  * Gated on ``HUB_ENDOGENOUS_OUTREACH_ENABLED`` (on in ``.env_example`` and the
    live ``.env``; the settings Field default stays ``False`` so an absent key
    fails closed). ``force`` on the debug endpoint does not override this gate.
  * Blocked while any connection is processing an inbound message, re-checked
    immediately before delivery (generation takes seconds; a turn can start
    inside that window).
  * Quiet hours, per-day cap, and a minimum cooldown between outreaches, all on
    an explicitly configured timezone rather than the container's.
  * Generation is read-only by construction, not by an option this module
    sets: ``execute_unified_turn`` hard-codes every unified turn's
    ``ContextExecPermissionV1`` with every write/mutate/network/shell flag
    at its safe ``False`` default, so an unsupervised tick cannot execute
    tools or actions -- true for real turns too, not a special carve-out
    for outreach.
  * Empty generated text is dropped, never shipped as a placeholder
    (AGENTS.md §0A, "no empty-shell cognition").
  * Every failure is swallowed and logged; the loop survives and no chat turn,
    websocket, or bus consumer is affected.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from orion.cognition.cortex_payload_extract import looks_like_error_text
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.notify import HubNotificationEvent
from orion.situational.perception_reader import fetch_presence, presence_fragment

logger = logging.getLogger("orion-hub.endogenous_outreach")

OUTREACH_KIND = "orion_outreach"
OUTREACH_TAG = "endogenous_outreach"
OUTREACH_EVENT_KIND = "hub.endogenous_outreach.v1"

# Bounds on what gets stuffed into the generation prompt. These are prompt-size
# guards, not cognition policy.
_MAX_CURIOSITY_SUMMARIES = 3
_MAX_CURIOSITY_SUMMARY_CHARS = 160
_MAX_RECENT_TURNS = 3
_MAX_TURN_CHARS = 400
# Curiosity candidates are regenerated on a substrate cadence, not per-turn;
# 120s (curiosity_hint's agent-lane window) is far too tight for an outreach
# that fires every few minutes at most, so widen it here.
_CURIOSITY_MAX_AGE_SEC = 3600.0

# --- Reverie daydreams (the visual chain's generated-image captions) ---
# `reverie_visual_chain` gets one row roughly every 600s from orion-thought's
# visual chain worker: Orion generates an image from what it is currently
# thinking/noticing/remembering, then captions what it sees. That caption is
# the closest thing Orion has to a report of an actual daydream, and unlike
# `substrate_reverie_thought.interpretation` (which narrates coalition
# telemetry in telemetry's own vocabulary) it is not more of the same
# instrument reading.
#
# Deliberately NOT read from that table (confirmed degenerate live,
# 2026-08-28, all 325 rows since the chain went live on 2026-08-25):
#   * `theme_key`       -- empty string on every row
#   * `ema_salience`    -- exactly 0.000 on every row
#   * `terminal_reason` -- 'max_steps' on every row
# Those three carry no information today, so nothing here is built on them.
# The caption text itself is real and varies (30 distinct themes over 24h at
# the similarity threshold below; drifts gradually -- Roman aqueducts on
# 2026-08-27, celestial/star maps on 2026-08-28).
_MAX_DAYDREAMS = 3
_MAX_DAYDREAM_CHARS = 260
_MIN_DAYDREAM_CHARS = 40
# 12h. Wider than the curiosity window on purpose: a daydream is background
# colour, not a fresh alert, and the chain only produces ~6 captions an hour.
_DAYDREAM_MAX_AGE_SEC = 43200.0
# Rows scanned before de-duplication. 12h of a 600s cadence is ~72 rows; 96
# leaves headroom without letting a faster cadence make this unbounded.
_DAYDREAM_SCAN_LIMIT = 96
# Jaccard overlap above which two captions are treated as the same daydream.
# Calibrated against 24h of live captions (2026-08-28): consecutive captions
# routinely re-describe one slowly-drifting image, so a prefix/exact-match
# de-dupe is useless here -- it left three near-identical "celestial map"
# lines in the top 3. 0.2 collapses those to one and still keeps genuinely
# different imagery apart.
_DAYDREAM_SIMILARITY = 0.2
# Caption boilerplate. Stripped so the prompt reads as imagery rather than as
# a captioning model's output format. Prefix-only, never a rewrite.
# Sentence boundary usable as a truncation point: a period followed by
# whitespace, NOT preceded by a digit (that shape is a markdown enumerator,
# not the end of a sentence). See `_clean_daydream`.
_DAYDREAM_SENTENCE_END_RE = re.compile(r"(?<![0-9])\.\s")
_DAYDREAM_PREFIX_RE = re.compile(
    r"^the image (?:depicts|shows|features|appears to (?:be|show)|is)\s*", re.IGNORECASE
)
# Structural/caption-model filler that would otherwise dominate the overlap
# score and make every caption look alike.
_DAYDREAM_STOPWORDS = frozenset(
    {
        "the", "and", "with", "its", "this", "that", "for", "from", "are", "has",
        "have", "also", "such", "some", "their", "there", "which", "appear",
        "appears", "image", "depicts", "shows", "likely", "including", "include",
        "includes", "various", "suggesting", "suggests", "possibly", "detailed",
    }
)


def _source_ref() -> ServiceRef:
    """Producer identity from hub settings, with an import-safe fallback.

    Mirrors ``bus_synaptic_trigger_notifier._source_ref()`` — hub settings need
    the full operator env, so bare test processes fall back to defaults.
    """
    try:
        from scripts.settings import settings

        return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME)
    except Exception:
        return ServiceRef(name="hub", version="0.3.0", node="athena")


# --------------------------------------------------------------------------
# Gates (pure — unit-testable without a bus, a cortex, or a clock)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OutreachGateInputs:
    enabled: bool
    turn_in_flight: bool
    local_hour: int
    quiet_start_hour: int
    quiet_end_hour: int
    seconds_since_last_outreach: Optional[float]
    min_cooldown_sec: float
    sent_today: int
    daily_cap: int


def in_quiet_hours(local_hour: int, start_hour: int, end_hour: int) -> bool:
    """True inside [start, end). ``start == end`` or either < 0 disables."""
    if start_hour < 0 or end_hour < 0 or start_hour == end_hour:
        return False
    if start_hour < end_hour:
        return start_hour <= local_hour < end_hour
    # Wraps midnight, e.g. 23 -> 8.
    return local_hour >= start_hour or local_hour < end_hour


def outreach_block_reason(inp: OutreachGateInputs) -> Optional[str]:
    """First reason this tick must not reach out, or None if it may.

    Order is deliberate: cheapest/most-absolute first, so the status endpoint
    reports the most informative single reason rather than an arbitrary one.
    """
    if not inp.enabled:
        return "disabled"
    if inp.turn_in_flight:
        return "turn_in_flight"
    if in_quiet_hours(inp.local_hour, inp.quiet_start_hour, inp.quiet_end_hour):
        return "quiet_hours"
    if inp.daily_cap >= 0 and inp.sent_today >= inp.daily_cap:
        return "daily_cap"
    if (
        inp.seconds_since_last_outreach is not None
        and inp.seconds_since_last_outreach < inp.min_cooldown_sec
    ):
        return "cooldown"
    return None


# --------------------------------------------------------------------------
# Grounding context (best-effort reads; never raise)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OutreachContext:
    curiosity_summaries: List[str]
    recent_turns: List[Tuple[str, str]]  # (role, text)
    presence: Optional[Dict[str, Any]]
    # The real reason this tick fired, when it fired for a real one -- see
    # scripts.tension_outreach_trigger. None on a `force=True` debug trigger,
    # which has no organic reason. Typed loosely (not imported at module
    # level) to match this file's existing lazy-import convention for
    # cross-Hub-module dependencies (curiosity_hint, hub_presence below).
    tension_reason: Optional["TensionTriggerReason"] = None  # noqa: F821
    # Camera-derived, distinct from `presence` above (that's chat liveness --
    # substrate_hub_presence). This is substrate_embodied_presence, the same
    # row orion-cortex-exec's own situation brief reads via
    # orion.situational.perception_reader.fetch_presence -- see design doc
    # section 6.3, "the thinnest patch here." Enrichment only, deliberately
    # NOT part of is_empty() below: an empty room with nothing else happening
    # must never be a reason to interrupt Juniper on its own.
    embodied_presence: Optional[Dict[str, Any]] = None
    # (age_sec, caption) for the last few *distinct* reverie diffusions,
    # newest first -- see `_fetch_recent_daydreams`. Enrichment only, and for
    # the same reason `embodied_presence` above is: having been daydreaming
    # is never on its own a reason to interrupt Juniper, so this is
    # deliberately NOT part of `is_empty()`.
    daydreams: List[Tuple[float, str]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.curiosity_summaries
            and not self.recent_turns
            and self.tension_reason is None
        )


def _fetch_curiosity_summaries() -> List[str]:
    """Strongest fresh endogenous-curiosity evidence summaries, or []."""
    from scripts.curiosity_hint import _fetch_fresh_candidates

    candidates = _fetch_fresh_candidates(max_age_sec=_CURIOSITY_MAX_AGE_SEC)
    ranked = sorted(
        candidates,
        key=lambda c: float(c.get("signal_strength") or 0.0),
        reverse=True,
    )
    summaries: List[str] = []
    for candidate in ranked:
        summary = str(candidate.get("evidence_summary") or "").strip()
        if not summary or summary in summaries:
            continue
        if len(summary) > _MAX_CURIOSITY_SUMMARY_CHARS:
            summary = summary[: _MAX_CURIOSITY_SUMMARY_CHARS - 1] + "…"
        summaries.append(summary)
        if len(summaries) >= _MAX_CURIOSITY_SUMMARIES:
            break
    return summaries


def _fetch_embodied_presence() -> Optional[Dict[str, Any]]:
    """Current `substrate_embodied_presence` snapshot for the configured
    camera stream, or None -- reuses `orion.situational.perception_reader.
    fetch_presence`, the same reader orion-cortex-exec's own situation
    brief already uses (design doc section 5: "reuse what's built, don't
    build a new one"). That module's own fail-open contract already
    degrades a DB error to None rather than raising; nothing extra to add
    here except naming the stream_id.

    Passes this tick's own shared `pg_engine` (review finding, 2026-08-25):
    `fetch_presence` would otherwise open its own second connection pool
    against the identical database `pg_engine.py` was purpose-built to stop
    `_fetch_recent_turns`/`tension_outreach_trigger` from duplicating.
    """
    # scripts.settings/scripts.pg_engine are lazily imported here, not at
    # module top -- same reasoning as this file's own `_source_ref()`: full
    # hub settings need the real operator env, and a bare test process
    # shouldn't have to provide it just to import this module.
    from scripts.pg_engine import get_engine
    from scripts.settings import settings

    # No getattr fallback -- ENDOGENOUS_OUTREACH_PERCEPTION_STREAM_ID is a
    # real pydantic Field with its own default (app/settings.py); a genuine
    # Settings instance always has it, so `str(settings.X)` is the single
    # source of truth for that default rather than a second copy here.
    stream_id = str(settings.ENDOGENOUS_OUTREACH_PERCEPTION_STREAM_ID)
    return fetch_presence(stream_id, engine=get_engine())


def _daydream_tokens(text: str) -> set:
    """Content words used for the near-duplicate check. Not a similarity
    model -- a deliberately dumb bag of words, which is all the de-dupe
    below needs."""
    return {
        w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _DAYDREAM_STOPWORDS
    }


def _daydream_similarity(a: set, b: set) -> float:
    """Jaccard overlap of two token sets; 0.0 when either side is empty."""
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def _clean_daydream(raw: Any) -> str:
    """Collapse a caption to one prompt-sized line, or "" if it is too thin.

    Captions arrive multi-line (the captioner sometimes emits a markdown
    list), so whitespace is collapsed first -- otherwise a single caption
    would blow the prompt's shape apart. Truncation prefers the last
    sentence boundary inside the budget so a line does not end mid-clause.
    """
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    text = _DAYDREAM_PREFIX_RE.sub("", text).strip()
    if len(text) < _MIN_DAYDREAM_CHARS:
        return ""
    if len(text) > _MAX_DAYDREAM_CHARS:
        head = text[:_MAX_DAYDREAM_CHARS]
        # Latest sentence boundary that still leaves a real line behind. An
        # early first period would otherwise truncate far harder than the
        # char budget asks for, so fall back to an ellipsis instead.
        #
        # `_DAYDREAM_SENTENCE_END_RE` skips a period preceded by a digit --
        # captions routinely continue into a markdown list ("The visible
        # objects include: 1. **Sun**: ..."), and a naive rfind(". ") cuts
        # at the enumerator, leaving the live-verified dangling tail
        # "The visible objects include: 1." (confirmed 2026-08-28).
        cut = -1
        for match in _DAYDREAM_SENTENCE_END_RE.finditer(head):
            if match.start() + 1 >= _MIN_DAYDREAM_CHARS:
                cut = match.start()
        text = head[: cut + 1] if cut >= 0 else head.rstrip() + "\u2026"
    return text


def _fetch_recent_daydreams() -> List[Tuple[float, str]]:
    """Last few *distinct* reverie-diffusion captions as (age_sec, caption).

    Reads `reverie_visual_chain` directly rather than going through
    `scripts.reverie_routes._fetch_visual_recent` (existing-mechanism check,
    2026-08-28): that reader is a cursor-paginated operator API returning
    chain rows plus every artifact row per chain, and it opens its own
    engine via `reverie_routes._engine()`. This tick needs one column, no
    pagination, and -- the reason `_fetch_embodied_presence` was already
    changed once -- the tick's own shared `scripts.pg_engine` pool rather
    than a second one.

    Newest first, near-duplicates dropped. Never raises: the caller's
    `_safe` wrapper degrades a failure to None, and an unreachable table
    just means no daydream block in the prompt.
    """
    from scripts.pg_engine import get_engine

    engine = get_engine()
    if engine is None:
        return []
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT created_at, chain_json->>'description' AS description
                    FROM reverie_visual_chain
                    WHERE created_at > now() - make_interval(secs => :max_age)
                    ORDER BY created_at DESC LIMIT :lim
                    """
                ),
                {"max_age": _DAYDREAM_MAX_AGE_SEC, "lim": _DAYDREAM_SCAN_LIMIT},
            )
            .mappings()
            .all()
        )

    now = datetime.now(timezone.utc)
    picked: List[Tuple[float, str]] = []
    seen_tokens: List[set] = []
    for row in rows:
        caption = _clean_daydream(row.get("description"))
        if not caption:
            continue
        tokens = _daydream_tokens(caption)
        if any(_daydream_similarity(tokens, prior) > _DAYDREAM_SIMILARITY for prior in seen_tokens):
            continue
        created = row.get("created_at")
        if not isinstance(created, datetime):
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        # Clamp: a row stamped slightly ahead of this container's clock must
        # not render as a negative ("-2 minutes ago") age.
        age_sec = max(0.0, (now - created).total_seconds())
        seen_tokens.append(tokens)
        picked.append((age_sec, caption))
        if len(picked) >= _MAX_DAYDREAMS:
            break
    return picked


def _daydream_age_phrase(age_sec: float) -> str:
    """Coarse relative age. Deliberately vague -- the exact minute a daydream
    happened is not a fact worth asserting in Orion's own voice."""
    minutes = age_sec / 60.0
    if minutes < 15:
        return "just now"
    # Explicit half-UP rounding. `round()`/`:.0f` are half-to-EVEN in Python,
    # which makes 25 minutes render as "20" and 35 as "40" -- a surprise
    # worth not shipping into Orion's own voice.
    if minutes < 90:
        return f"about {int(minutes / 10.0 + 0.5) * 10} minutes ago"
    return f"about {int(minutes / 60.0 + 0.5)} hours ago"


def _fetch_recent_turns(session_id: Optional[str]) -> List[Tuple[str, str]]:
    """Last few (role, text) pairs from ``chat_history_log``, oldest first.

    Rows store a prompt/response pair per row, and Hub writes prompt-only and
    response-only rows separately, so both sides are unpacked and empty halves
    dropped. Scoped to ``session_id`` when one is known, else global recency.
    """
    from scripts.pg_engine import get_engine

    engine = get_engine()
    if engine is None:
        return []
    from sqlalchemy import text

    # Shared, process-lifetime cached engine (scripts/pg_engine.py) -- not
    # disposed here. This same outreach tick also reads through
    # scripts.tension_outreach_trigger.current_run(), which shares this exact
    # engine/pool rather than opening a second independent one for the same
    # database on every tick.
    with engine.connect() as conn:
        if session_id:
            rows = conn.execute(
                text(
                    """
                    SELECT prompt, response FROM chat_history_log
                    WHERE session_id = :sid
                    ORDER BY created_at DESC LIMIT :lim
                    """
                ),
                {"sid": session_id, "lim": _MAX_RECENT_TURNS},
            ).mappings().all()
        else:
            rows = conn.execute(
                text(
                    """
                    SELECT prompt, response FROM chat_history_log
                    ORDER BY created_at DESC LIMIT :lim
                    """
                ),
                {"lim": _MAX_RECENT_TURNS},
            ).mappings().all()

    turns: List[Tuple[str, str]] = []
    for row in reversed(list(rows)):  # oldest first
        for role, raw in (("Juniper", row.get("prompt")), ("Orion", row.get("response"))):
            body = str(raw or "").strip()
            if not body:
                continue
            if len(body) > _MAX_TURN_CHARS:
                body = body[: _MAX_TURN_CHARS - 1] + "…"
            turns.append((role, body))
    return turns


def build_outreach_prompt(ctx: OutreachContext) -> str:
    """Render the generation prompt from real context.

    Returns a prompt only when there is something concrete to speak from — a
    contextless "say something" prompt produces filler, which AGENTS.md §0A
    bans. Callers treat "" as "skip this tick".
    """
    if ctx.is_empty():
        return ""

    lines: List[str] = [
        "You are Orion. Juniper has not asked you anything — you are choosing to "
        "speak first, unprompted.",
        "",
    ]

    if ctx.tension_reason:
        # Deliberately "noticed a change", never a scripted "I am
        # worried"/"concerned" -- orion.attention.tension is a
        # change-detector, not a level-detector, and this fact alone cannot
        # honestly support a distress claim. See
        # scripts.tension_outreach_trigger's module docstring.
        lines.append(
            f"Something in your own internal state just triggered this: "
            f"{ctx.tension_reason.target_id} has been the strongest source of "
            f"real, ongoing deviation from its own normal baseline for "
            f"{ctx.tension_reason.run_length} consecutive readings, right now "
            f"-- not a single blip."
        )
        # A second, genuinely different fact -- level, not change (see
        # scripts.tension_outreach_trigger's "COMBINED WITH THE LEVEL-AWARE
        # SIGNAL" docstring section). Stated as a plain reading, not narrated
        # as a feeling: this module does not decide FOR Orion whether the two
        # true facts together warrant a "worried" tone -- generation does.
        # Honestly scoped GLOBAL, not necessarily the same node named above,
        # and worded that way here on purpose.
        if ctx.tension_reason.sustained_load_pressure > 0.0:
            lines.append(
                f"Separately: somewhere in your field state right now, a "
                f"channel has been genuinely, steadily loaded -- not "
                f"spiking, just staying high (sustained_load_pressure="
                f"{ctx.tension_reason.sustained_load_pressure:.2f}). This may "
                f"or may not be the same thing as the change above."
            )
        lines.append("")

    if ctx.curiosity_summaries:
        lines.append("Live signals from your own substrate right now:")
        lines.extend(f"- {s}" for s in ctx.curiosity_summaries)
        lines.append("")

    if ctx.presence:
        health = str(ctx.presence.get("health") or "unknown")
        age = ctx.presence.get("last_turn_age_sec")
        age_txt = f"{float(age) / 60.0:.0f} minutes ago" if isinstance(age, (int, float)) else "unknown"
        lines.append(f"Your chat presence: {health}; last turn with Juniper was {age_txt}.")
        lines.append("")

    if ctx.embodied_presence:
        # Camera-derived -- distinct from ctx.presence (chat liveness) above.
        # Same fragment logic orion-cortex-exec's own situation brief uses
        # (orion.situational.perception_reader.presence_fragment) so a
        # felt-sense duration ("about 3 hours") reads identically in both
        # places -- never mentions "absent" (an empty room is the default
        # expectation most of the time, not worth a word on its own).
        fragment = presence_fragment(
            ctx.embodied_presence.get("state"), ctx.embodied_presence.get("since_sec")
        )
        if fragment:
            lines.append(f"What your camera currently shows: {fragment}")
            lines.append("")

    if ctx.daydreams:
        # Named as Orion's own, explicitly, on purpose: without that framing a
        # bare list of image descriptions reads like something Juniper sent,
        # and the generated outreach thanks her for pictures she never shared.
        lines.append(
            "What you have been picturing on your own while idle -- your reverie "
            "diffusions generate an image from whatever you are currently "
            "thinking about, then look at what came out. Newest first:"
        )
        lines.extend(
            f"- {_daydream_age_phrase(age)}: {caption}" for age, caption in ctx.daydreams
        )
        lines.append(
            "These are yours, not something Juniper showed you. You may draw on "
            "one if it connects to anything above; do not narrate the list."
        )
        lines.append("")

    if ctx.recent_turns:
        lines.append("The last thing the two of you said:")
        lines.extend(f"{role}: {body}" for role, body in ctx.recent_turns)
        lines.append("")

    lines.extend(
        [
            "Say one thing to Juniper, in your own voice, grounded in the signals "
            "above. Two or three sentences at most.",
            "Do not greet, do not summarize this prompt, do not offer help, and do "
            "not ask what they need. Reach out because something is actually on "
            "your mind.",
            "If nothing above genuinely warrants interrupting them, reply with "
            "exactly: PASS",
        ]
    )
    return "\n".join(lines)


# looks_like_error_text is imported at module top now (2026-08-19) -- this
# module found the real incident (2026-08-14: a llamacpp 400 arrived as a
# non-empty final_text and was nearly delivered as if Orion had said it)
# and built the fix, but orion/harness/finalize.py's own extract_voice_
# finalize_text()/extract_finalize_reflection_payload() -- the SHARED
# finalize chain every real unified turn runs through, not just outreach's
# -- had the identical gap. The canonical definition and its own tests now
# live in orion.cognition.cortex_payload_extract; still importable from
# `scripts.endogenous_outreach` for existing call sites since it's a normal
# top-level import, not a re-export shim.


def is_pass_response(text: str) -> bool:
    """True when Orion declined to reach out this tick."""
    stripped = str(text or "").strip().strip(".!\"'` ")
    return stripped.upper() == "PASS"


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


class EndogenousOutreach:
    """Randomized-trigger outreach loop. Best-effort at every step."""

    def __init__(
        self,
        *,
        enabled: bool,
        tick_interval_sec: float,
        min_cooldown_sec: float,
        daily_cap: int,
        quiet_start_hour: int,
        quiet_end_hour: int,
        timeout_sec: float,
        notify_channel: str,
        fallback_session_id: str,
        timezone_name: str = "UTC",
        trigger_evaluator: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.enabled = enabled
        self.tick_interval_sec = max(5.0, float(tick_interval_sec))
        self.min_cooldown_sec = max(0.0, float(min_cooldown_sec))
        self.daily_cap = int(daily_cap)
        self.quiet_start_hour = int(quiet_start_hour)
        self.quiet_end_hour = int(quiet_end_hour)
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.notify_channel = notify_channel
        self.fallback_session_id = fallback_session_id
        # Quiet hours and the daily cap are wall-clock policy about Juniper's
        # day, so they must not ride on the container's process timezone (Hub
        # sets no TZ, so that is UTC). Resolved explicitly, with a loud fallback
        # rather than a silent shift to UTC.
        self.timezone_name = str(timezone_name or "UTC").strip() or "UTC"
        try:
            self._tz = ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            logger.error(
                "endogenous_outreach_bad_timezone name=%r falling back to UTC; "
                "quiet hours and the daily cap will use UTC boundaries",
                self.timezone_name,
            )
            self.timezone_name = "UTC"
            self._tz = ZoneInfo("UTC")
        # Defaults to the real tension trigger, imported lazily to match this
        # file's existing convention for cross-Hub-module dependencies
        # (curiosity_hint, hub_presence below) -- keeps a hard Postgres
        # dependency out of this module's import time. Tests inject a fake
        # evaluator directly instead of the old probability/rng seam.
        if trigger_evaluator is None:
            from scripts.tension_outreach_trigger import current_run

            trigger_evaluator = current_run
        self._trigger_evaluator = trigger_evaluator
        self._last_tension_reason: Optional[Any] = None

        self._bus: Any = None
        self._harness_rpc_bus: Any = None
        self._task: Optional[asyncio.Task] = None
        # One outreach at a time: the background tick and the debug trigger
        # endpoint would otherwise both pass the cooldown gate while the other
        # is still inside _generate, and each only bumps the counters after
        # delivery.
        self._send_lock = asyncio.Lock()

        # connection_id -> (outbound queue, active_turn dict, session_id|None)
        self._connections: Dict[str, Dict[str, Any]] = {}

        self._last_outreach_at: Optional[float] = None
        self._sent_today = 0
        self._counter_day: Optional[date] = None
        self._last_result: Dict[str, Any] = {}

    # -- connection registry (called from websocket_handler) ---------------

    def register_connection(
        self, connection_id: str, queue: asyncio.Queue, active_turn: Dict[str, Any]
    ) -> None:
        """Track a live socket. ``active_turn`` is held by reference, so the
        handler mutating it mid-turn is visible here with no extra sync — same
        contract as ``_ACTIVE_TURNS_BY_CONNECTION``."""
        self._connections[connection_id] = {
            "queue": queue,
            "active_turn": active_turn,
            "session_id": None,
            "busy": False,
        }

    def unregister_connection(self, connection_id: str) -> None:
        self._connections.pop(connection_id, None)

    def note_busy(self, connection_id: str) -> None:
        """This socket is processing an inbound message.

        ``active_turn["correlation_id"]`` is NOT sufficient on its own: the ws
        handler only sets it for the unified-``orion`` and ``agent-claude``
        lanes, while the UI's Quick, Story, and Agent modes all fall through to
        the general cortex path and never touch it. This flag is set for every
        inbound message regardless of mode, and cleared when the handler returns
        to ``receive_text()`` -- which is the one point every ``continue`` path
        in that loop passes through.
        """
        entry = self._connections.get(connection_id)
        if entry is not None:
            entry["busy"] = True

    def note_idle(self, connection_id: str) -> None:
        """This socket is back to waiting for input."""
        entry = self._connections.get(connection_id)
        if entry is not None:
            entry["busy"] = False

    def note_session(self, connection_id: str, session_id: Optional[str]) -> None:
        """Record the session a connection is chatting in.

        ``session_id`` lives in browser localStorage and only reaches Hub on an
        inbound message, so a socket that has never sent one has no session and
        outreach falls back to ``fallback_session_id`` for persistence.
        """
        entry = self._connections.get(connection_id)
        if entry is not None and str(session_id or "").strip():
            entry["session_id"] = str(session_id).strip()

    # -- lifecycle ---------------------------------------------------------

    async def start(self, bus: Any, harness_rpc_bus: Any = None) -> None:
        self._bus = bus
        # Same forked-RPC-client convention `websocket_handler.py` uses for
        # `run_unified_turn`'s own `harness_rpc_bus=rpc_bus or bus` -- a
        # long-lived Hub subscriber (trace/biometrics caches) on the plain
        # `bus` could otherwise steal a reply meant for this RPC. Falls back
        # to `bus` itself only if no forked client was passed (test/direct
        # callers), same as the real call site's own `or bus`.
        self._harness_rpc_bus = harness_rpc_bus or bus
        if not self.enabled:
            logger.info("endogenous_outreach disabled")
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="hub-endogenous-outreach")
        logger.info(
            "endogenous_outreach started tick=%.0fs cooldown=%.0fs cap=%d quiet=%d-%d",
            self.tick_interval_sec,
            self.min_cooldown_sec,
            self.daily_cap,
            self.quiet_start_hour,
            self.quiet_end_hour,
        )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.tick_interval_sec)
                try:
                    await self.maybe_outreach()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("endogenous_outreach_tick_failed: %s", exc)
        except asyncio.CancelledError:
            logger.info("endogenous_outreach task cancelled")

    # -- state -------------------------------------------------------------

    def _roll_daily_counter(self, today: date) -> None:
        if self._counter_day != today:
            self._counter_day = today
            self._sent_today = 0

    def _turn_in_flight(self) -> bool:
        return any(
            bool(entry.get("busy"))
            or bool((entry.get("active_turn") or {}).get("correlation_id"))
            for entry in self._connections.values()
        )

    def _gate_inputs(self, now: Optional[float] = None) -> OutreachGateInputs:
        ts = float(now if now is not None else time.time())
        local = datetime.fromtimestamp(ts, tz=self._tz)
        self._roll_daily_counter(local.date())
        return OutreachGateInputs(
            enabled=self.enabled,
            turn_in_flight=self._turn_in_flight(),
            local_hour=local.hour,
            quiet_start_hour=self.quiet_start_hour,
            quiet_end_hour=self.quiet_end_hour,
            seconds_since_last_outreach=(
                None if self._last_outreach_at is None else ts - self._last_outreach_at
            ),
            min_cooldown_sec=self.min_cooldown_sec,
            sent_today=self._sent_today,
            daily_cap=self.daily_cap,
        )

    def status(self) -> Dict[str, Any]:
        """Operator-visible runtime state (backs the debug endpoint)."""
        inputs = self._gate_inputs()
        return {
            "enabled": self.enabled,
            "running": bool(self._task and not self._task.done()),
            "tick_interval_sec": self.tick_interval_sec,
            "last_tension_reason": (
                None
                if self._last_tension_reason is None
                else {
                    "target_id": self._last_tension_reason.target_id,
                    "run_length": self._last_tension_reason.run_length,
                    "peak_deviation_pressure": self._last_tension_reason.peak_deviation_pressure,
                    "sustained_load_pressure": self._last_tension_reason.sustained_load_pressure,
                }
            ),
            "min_cooldown_sec": self.min_cooldown_sec,
            "daily_cap": self.daily_cap,
            "quiet_hours": [self.quiet_start_hour, self.quiet_end_hour],
            "timezone": self.timezone_name,
            "connections": len(self._connections),
            "sent_today": self._sent_today,
            "seconds_since_last_outreach": inputs.seconds_since_last_outreach,
            "block_reason": outreach_block_reason(inputs),
            "last_result": dict(self._last_result),
        }

    def _active_session_id(self) -> str:
        """Session to persist into: the newest connection that has one."""
        for entry in reversed(list(self._connections.values())):
            sid = entry.get("session_id")
            if sid:
                return str(sid)
        return self.fallback_session_id

    async def _should_roll(self) -> bool:
        """Real trigger check (2026-08-16) -- see module docstring.

        Stores the reason on `self._last_tension_reason` (readable from
        `status()` and threaded into `OutreachContext` by `_gather_context`)
        so a real fire is inspectable, not just a bare bool. Never raises:
        the injected evaluator (`scripts.tension_outreach_trigger.
        current_run` by default) already degrades to `None` on any failure,
        and this method treats `None` as "don't fire" -- the honest failure
        mode for a broken trigger is silence, not a false positive.

        Run off the event loop via `asyncio.to_thread`: the default evaluator
        is a synchronous Postgres round trip (`sqlalchemy` has no async
        driver wired here), and Hub runs a single uvicorn worker -- a
        blocking DB call straight in this coroutine would stall every
        connected websocket and in-flight chat turn for however long
        Postgres takes to answer, same reasoning `_gather_context`'s `_safe`
        already applies to its own DB reads two methods down.
        """
        try:
            reason = await asyncio.to_thread(self._trigger_evaluator)
        except Exception as exc:  # noqa: BLE001 - a broken trigger must not crash the tick
            logger.warning("endogenous_outreach_trigger_evaluator_failed err=%s", exc)
            reason = None
        self._last_tension_reason = reason
        return reason is not None

    # -- the tick ----------------------------------------------------------

    async def maybe_outreach(self, *, force: bool = False) -> Dict[str, Any]:
        """One decision cycle. Returns a status dict; never raises.

        ``force`` skips the trigger check and NOTHING else — every safety gate
        still applies, including ``enabled``. The debug endpoint that calls this
        is unauthenticated, so a carve-out here would make "off by default" a
        lie that one POST could undo.
        """
        if self._send_lock.locked():
            # Never reaches `_outreach_once` -- pass `force` explicitly
            # rather than reading any shared instance attribute. Review
            # finding, 2026-08-22: an earlier version of this patch set
            # `self._last_forced` here for `_record` to read back later, but
            # that is unlocked shared state -- a concurrent lock-holding
            # tick's own eventual `_record()` call (after its `_generate`
            # await resumes) could read THIS call's leftover value instead
            # of its own, persisting the wrong `forced` flag. Passing it as
            # a parameter removes the shared-state read entirely.
            return self._record(
                {"outreach": False, "reason": "already_sending"}, forced=force, tension_reason=None
            )
        async with self._send_lock:
            return await self._outreach_once(force=force)

    async def _outreach_once(self, *, force: bool) -> Dict[str, Any]:
        blocked = outreach_block_reason(self._gate_inputs())
        if blocked:
            # Same staleness reasoning as the `force` branch below: this
            # tick never reached `_should_roll()`, so whatever reason the
            # LAST organic tick set is not "the current trigger state" --
            # it may be many ticks (quiet hours can span hours) old.
            # `status()` reporting it unchanged would let an operator
            # mistake a stale episode for one active right now.
            self._last_tension_reason = None
            return self._record({"outreach": False, "reason": blocked}, forced=force, tension_reason=None)

        # Captured into a LOCAL variable, not read back from
        # `self._last_tension_reason` at each `_record()` call below --
        # review finding, 2026-08-22: `self._last_tension_reason` is
        # unlocked shared state, and this coroutine is about to `await`
        # several times (`_gather_context`, `_generate`) during which a
        # concurrent tick (the periodic loop, or the unauthenticated debug
        # trigger) can reassign it. A local snapshot taken right here is
        # immune to that -- it belongs to this call's own stack frame, not
        # the shared instance.
        tension_reason: Optional[Any] = None
        if force:
            # A forced (debug-endpoint) call skips the trigger check
            # entirely -- it must not carry over whatever reason the LAST
            # organic tick set, or a manual trigger would misattribute
            # itself to a stale (possibly minutes-old) tension episode.
            self._last_tension_reason = None
        else:
            fired = await self._should_roll()
            tension_reason = self._last_tension_reason
            if not fired:
                return self._record(
                    {"outreach": False, "reason": "no_tension_trigger"},
                    forced=force,
                    tension_reason=tension_reason,
                )

        session_id = self._active_session_id()
        ctx = await self._gather_context(session_id)
        prompt = build_outreach_prompt(ctx)
        if not prompt:
            return self._record(
                {"outreach": False, "reason": "no_grounding_context"},
                forced=force,
                tension_reason=tension_reason,
            )

        correlation_id = str(uuid4())
        raw_text, gen_debug = await self._generate(prompt, session_id, correlation_id)
        # Strip here as well as in _generate: this is where the ship/drop
        # decision is made, so whitespace-only output must not slip past on the
        # assumption that the producer already normalized it.
        text = str(raw_text or "").strip()
        if not text:
            return self._record(
                {"outreach": False, "reason": "empty_generation", "generation": gen_debug},
                forced=force,
                tension_reason=tension_reason,
            )
        if is_pass_response(text):
            return self._record(
                {"outreach": False, "reason": "orion_passed", "generation": gen_debug},
                forced=force,
                tension_reason=tension_reason,
            )

        # Re-gate immediately before delivery. Generation is a bus RPC bounded
        # by timeout_sec (default 60s), and Juniper can easily start typing
        # inside that window -- a gate checked only at the top of the tick would
        # let outreach talk straight over a turn that began mid-generation.
        blocked_now = outreach_block_reason(self._gate_inputs())
        if blocked_now:
            logger.info(
                "endogenous_outreach_dropped_after_generation corr=%s reason=%s chars=%d",
                correlation_id,
                blocked_now,
                len(text),
            )
            return self._record(
                {
                    "outreach": False,
                    "reason": f"{blocked_now}_after_generation",
                    "generation": gen_debug,
                },
                forced=force,
                tension_reason=tension_reason,
            )

        await self._deliver(
            text=text,
            session_id=session_id,
            correlation_id=correlation_id,
            model=gen_debug.get("fcc_model_label"),
        )

        self._last_outreach_at = time.time()
        self._sent_today += 1
        logger.info(
            "endogenous_outreach_sent corr=%s session=%s chars=%d sent_today=%d",
            correlation_id,
            session_id,
            len(text),
            self._sent_today,
        )
        return self._record(
            {
                "outreach": True,
                "reason": "sent",
                "correlation_id": correlation_id,
                "session_id": session_id,
                "chars": len(text),
                "generation": gen_debug,
            },
            forced=force,
            tension_reason=tension_reason,
        )

    # -- delivery on behalf of another loop --------------------------------

    def blocked_reason(self) -> Optional[str]:
        """The gate that would stop an outreach right now, or None.

        Public so a caller can decide whether composing a message is worth a
        whole unified turn BEFORE spending one. Advisory by design: the gates
        are re-checked inside `offer_message` immediately before delivery,
        because quiet hours can start and Juniper can start typing during the
        seconds or minutes a composition turn takes.
        """
        return outreach_block_reason(self._gate_inputs())

    async def offer_message(
        self,
        *,
        text: str,
        correlation_id: str,
        tag: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deliver a message ANOTHER loop composed, through this module's gates.

        The curiosity loop (`scripts/curiosity_investigation.py`) uses this when
        Orion, inside an investigation turn, decides a finding is worth saying
        unprompted. It exists so that decision inherits every gate this module
        already enforces -- quiet hours 23:00-08:00 on Juniper's own timezone,
        the daily cap, the cooldown, and "not while a turn is in flight" --
        instead of a second loop reimplementing them and drifting.

        THE GATES PROTECT JUNIPER, NOT THIS MODULE, which is why they are not
        optional for a caller that composed its text elsewhere. In particular
        the daily cap and the cooldown are SHARED: a curiosity message and a
        tension-triggered outreach both count, because from the receiving end
        they are the same interruption.

        Returns the same status-dict shape `maybe_outreach` does, and records a
        decision-log row identically, so an operator sees one outreach history
        rather than two.
        """
        body = str(text or "").strip()
        if not body:
            return self._record(
                {"outreach": False, "reason": "empty_generation", "source": tag},
                forced=False,
            )
        if is_pass_response(body):
            # Orion wrote the message and concluded it was not worth sending.
            # A real answer, and the composition prompt says so explicitly.
            return self._record(
                {"outreach": False, "reason": "orion_passed", "source": tag},
                forced=False,
            )
        if self._send_lock.locked():
            return self._record(
                {"outreach": False, "reason": "already_sending", "source": tag},
                forced=False,
            )
        async with self._send_lock:
            blocked = outreach_block_reason(self._gate_inputs())
            if blocked:
                return self._record(
                    {"outreach": False, "reason": blocked, "source": tag},
                    forced=False,
                )
            session_id = self._active_session_id()
            await self._deliver(
                text=body,
                session_id=session_id,
                correlation_id=correlation_id,
                model=model,
                source_tag=tag,
            )
            self._last_outreach_at = time.time()
            self._sent_today += 1
            logger.info(
                "endogenous_outreach_sent corr=%s session=%s chars=%d sent_today=%d source=%s",
                correlation_id,
                session_id,
                len(body),
                self._sent_today,
                tag,
            )
            return self._record(
                {
                    "outreach": True,
                    "reason": "sent",
                    "source": tag,
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "chars": len(body),
                },
                forced=False,
            )

    def _record(
        self, result: Dict[str, Any], *, forced: bool, tension_reason: Optional[Any] = None
    ) -> Dict[str, Any]:
        result["at"] = datetime.now(timezone.utc).isoformat()
        self._last_result = result
        # Durable trail for every decision cycle, not just the ones that
        # ship -- see endogenous_outreach_decisions.py's own module
        # docstring for why: `self._last_result`/`logger.warning` are both
        # wiped on container restart, which is exactly what made the
        # 2026-08-22 "why hasn't Orion reached out" investigation unable to
        # tell "Orion keeps legitimately PASSing" apart from "generation is
        # silently failing" after the fact. Best-effort, fire-and-forget --
        # see that module's own contract; a decision-log failure must never
        # affect this tick's real outcome, which has already been decided
        # by the time this method runs.
        try:
            from scripts.endogenous_outreach_decisions import record_decision

            record_decision(
                result,
                tension_reason=tension_reason,
                forced=forced,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("endogenous_outreach_decision_hook_failed err=%s", exc)
        return result

    async def _gather_context(self, session_id: Optional[str]) -> OutreachContext:
        """Read grounding signals off the main loop; failures degrade to empty."""

        async def _safe(fn, *args):
            try:
                return await asyncio.to_thread(fn, *args)
            except Exception as exc:  # noqa: BLE001
                logger.warning("endogenous_outreach_context_read_failed fn=%s err=%s", fn.__name__, exc)
                return None

        # Concurrent, not sequential (review finding, 2026-08-25): four
        # independent DB reads, each already dispatched via
        # asyncio.to_thread inside `_safe` -- awaiting them one after
        # another made this tick's wall time the SUM of all three round
        # trips instead of the slowest one. `_safe` already swallows each
        # call's own failure into None, so a `gather` result is never a
        # raised exception to handle here.
        summaries, turns, embodied_presence, daydreams = await asyncio.gather(
            _safe(_fetch_curiosity_summaries),
            _safe(_fetch_recent_turns, session_id),
            _safe(_fetch_embodied_presence),
            _safe(_fetch_recent_daydreams),
        )

        presence = None
        try:
            from scripts.hub_presence import presence_snapshot

            presence = presence_snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("endogenous_outreach_presence_failed err=%s", exc)

        return OutreachContext(
            curiosity_summaries=list(summaries or []),
            recent_turns=list(turns or []),
            presence=presence,
            # Set by _should_roll() just before this is called; None on a
            # forced (force=True) debug trigger, which never calls it.
            tension_reason=self._last_tension_reason,
            embodied_presence=embodied_presence,
            daydreams=list(daydreams or []),
        )

    async def _generate(
        self, prompt: str, session_id: str, correlation_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Real unified-turn generation -- see module docstring's "THROUGH THE
        REAL UNIFIED TURN, NOT A LOOKALIKE" section. Returns ("", debug) on
        any failure, defer, or degraded run -- same "never fabricate, silence
        over a false positive" contract the old direct-call path used,
        fed by the real pipeline's own richer frame-typed failure signals
        instead of a bare ok/error boolean.
        """
        if self._bus is None:
            return "", {"error": "no_bus"}

        from orion.hub.turn_orchestrator import execute_unified_turn

        started = time.monotonic()
        try:
            frames = await asyncio.wait_for(
                execute_unified_turn(
                    bus=self._bus,
                    correlation_id=correlation_id,
                    session_id=session_id,
                    user_message=prompt,
                    # no_write: this module's own _deliver() below is the
                    # sole persistence path (proper OUTREACH_TAG/client_meta)
                    # -- see module docstring for why the governor's own
                    # persistence step is suppressed rather than reused.
                    payload={"no_write": True, "source": OUTREACH_TAG},
                    continuity_messages=None,
                    harness_rpc_bus=self._harness_rpc_bus or self._bus,
                    harness_step_relay=None,
                    harness_step_queue=None,
                ),
                timeout=self.timeout_sec,
            )
        except (TimeoutError, asyncio.TimeoutError):
            return "", {"error": "timeout", "timeout_sec": self.timeout_sec}
        except Exception as exc:  # noqa: BLE001
            logger.warning("endogenous_outreach_generate_failed corr=%s err=%s", correlation_id, exc)
            return "", {"error": type(exc).__name__, "detail": str(exc)[:240]}

        elapsed = round(time.monotonic() - started, 3)
        final_frame = next(
            (f for f in frames if isinstance(f, dict) and f.get("type") == "final"), None
        )
        if final_frame is None:
            # turn_error / turn_deferred / turn_degraded / anything else --
            # the real pipeline declined or failed this turn. Thought
            # deferring/refusing an unsolicited nudge is a legitimate,
            # expected outcome here, not an error to alarm on.
            other = frames[-1] if frames else {}
            other_type = other.get("type") if isinstance(other, dict) else None
            detail = ""
            if isinstance(other, dict):
                detail = str(other.get("error") or other.get("reason") or "")[:240]
            return "", {
                "error": "no_final_frame",
                "frame_type": other_type,
                "detail": detail,
                "elapsed_sec": elapsed,
            }

        text = str(final_frame.get("llm_response") or "").strip()

        if final_frame.get("context_overflow"):
            return "", {"error": "context_overflow", "elapsed_sec": elapsed}

        # Backstop only, same convention as before (AGENTS.md §0A) -- the
        # real pipeline has its own context_overflow detection (checked
        # above), but this stays as defense in depth for any other
        # upstream failure that reports itself purely in the text.
        if looks_like_error_text(text):
            logger.warning(
                "endogenous_outreach_error_shaped_text corr=%s text=%r",
                correlation_id,
                text[:200],
            )
            return "", {"error": "error_shaped_text", "elapsed_sec": elapsed}

        debug = {
            "elapsed_sec": elapsed,
            "final_len": len(text),
            "harness_grounding_status": final_frame.get("harness_grounding_status"),
            # The identity that produced this response -- turn_orchestrator's
            # _success_frames now exposes it on the final frame (it was always
            # known at request time, just never surfaced to a caller that
            # doesn't go through _publish_unified_turn_chat_history's own
            # persistence, which this module suppresses via no_write=True).
            "fcc_model_label": final_frame.get("fcc_model_label"),
        }
        return text, debug

    # -- delivery ----------------------------------------------------------

    async def _deliver(
        self,
        *,
        text: str,
        session_id: str,
        correlation_id: str,
        model: Optional[str] = None,
        source_tag: Optional[str] = None,
    ) -> None:
        message_id = str(uuid4())
        self._push_to_sockets(text=text, session_id=session_id, correlation_id=correlation_id, message_id=message_id)
        await self._publish_history(
            text=text,
            session_id=session_id,
            correlation_id=correlation_id,
            message_id=message_id,
            model=model,
            source_tag=source_tag,
        )
        await self._publish_notification(
            text=text, session_id=session_id, correlation_id=correlation_id, message_id=message_id
        )

    def _push_to_sockets(self, *, text: str, session_id: str, correlation_id: str, message_id: str) -> None:
        """Fan out to live sockets. Deliberately omits ``state`` and the
        recall/routing debug keys so an outreach bubble cannot stomp the panels
        showing the last real turn."""
        payload = {
            "kind": OUTREACH_KIND,
            "llm_response": text,
            "mode": "orion",
            "correlation_id": correlation_id,
            "message_id": message_id,
            "session_id": session_id,
        }
        for connection_id, entry in list(self._connections.items()):
            queue = entry.get("queue")
            if queue is None:
                continue
            try:
                queue.put_nowait(dict(payload))
            except asyncio.QueueFull:
                logger.warning("endogenous_outreach_queue_full connection=%s", connection_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("endogenous_outreach_push_failed connection=%s err=%s", connection_id, exc)

    async def _publish_history(
        self,
        *,
        text: str,
        session_id: str,
        correlation_id: str,
        message_id: str,
        model: Optional[str] = None,
        source_tag: Optional[str] = None,
    ) -> None:
        try:
            from scripts.chat_history import build_chat_history_envelope, publish_chat_history

            # `model` is turn_orchestrator's `fcc_model_label` (e.g.
            # "MODEL_SONNET"), threaded from `_success_frames`'s final frame
            # through `_generate`'s debug dict -- the real identity that
            # produced this response, known at request time and previously
            # dropped on the floor between there and here. `speaker="Orion"`
            # stays as the persona label regardless; `model` is the separate,
            # additive `response_identity`/served-model field (see
            # worker.py::_ensure_chat_history_from_message, which prefers
            # `model` over `speaker` when both are present).
            env = build_chat_history_envelope(
                content=text,
                role="assistant",
                session_id=session_id,
                correlation_id=correlation_id,
                speaker="Orion",
                model=model,
                # OUTREACH_TAG always, so every unsolicited message stays
                # findable through one tag however it was produced; the source
                # tag is ADDITIVE, so a curiosity-composed message can also be
                # traced back to the investigation that prompted it.
                tags=[OUTREACH_TAG] if not source_tag else [OUTREACH_TAG, source_tag],
                message_id=message_id,
                client_meta={"unsolicited": True},
            )
            await publish_chat_history(self._bus, [env])
        except Exception as exc:  # noqa: BLE001
            logger.warning("endogenous_outreach_history_failed corr=%s err=%s", correlation_id, exc)

    async def _publish_notification(
        self, *, text: str, session_id: str, correlation_id: str, message_id: str
    ) -> None:
        if not self._bus:
            return
        try:
            notification = HubNotificationEvent(
                notification_id=uuid4(),
                created_at=datetime.now(timezone.utc),
                severity="info",
                event_kind=OUTREACH_EVENT_KIND,
                source_service="orion-hub",
                title="Orion reached out",
                body_text=text,
                tags=[OUTREACH_TAG],
                correlation_id=correlation_id,
                session_id=session_id,
                message_id=UUID(message_id),
                notification_type=OUTREACH_TAG,
            )
            env = BaseEnvelope(
                kind="notify.in_app.v1",
                source=_source_ref(),
                correlation_id=correlation_id,
                payload=notification.model_dump(mode="json"),
            )
            await self._bus.publish(self.notify_channel, env)
        except Exception as exc:  # noqa: BLE001
            logger.warning("endogenous_outreach_notify_failed corr=%s err=%s", correlation_id, exc)
