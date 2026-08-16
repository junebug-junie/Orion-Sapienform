"""Endogenous outreach — Orion opens a conversation Juniper did not start.

STUB TRIGGER, DELIBERATELY. Orion has no endogenous "I want to say something
now" signal yet (that is what the autonomy/drive work is building toward), so
this module fires on a randomized timer instead of on a real motivational
state. The timer is the *only* stubbed part: the message itself is generated
from live substrate signals and real chat history, and lands on the same three
rails a normal turn uses. When a real endogenous trigger exists, replace
``_should_roll()`` and delete nothing else.

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
  * Generation pins ``tool_execution_policy``/``action_execution_policy`` to
    ``none`` and disables recall, so an unsupervised tick cannot execute tools
    or actions.
  * Empty generated text is dropped, never shipped as a placeholder
    (AGENTS.md §0A, "no empty-shell cognition").
  * Every failure is swallowed and logged; the loop survives and no chat turn,
    websocket, or bus consumer is affected.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.notify import HubNotificationEvent

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

    def is_empty(self) -> bool:
        return not self.curiosity_summaries and not self.recent_turns


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


def _fetch_recent_turns(session_id: Optional[str]) -> List[Tuple[str, str]]:
    """Last few (role, text) pairs from ``chat_history_log``, oldest first.

    Rows store a prompt/response pair per row, and Hub writes prompt-only and
    response-only rows separately, so both sides are unpacked and empty halves
    dropped. Scoped to ``session_id`` when one is known, else global recency.
    """
    import os

    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return []
    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    try:
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
    finally:
        engine.dispose()

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


_ERROR_TEXT_PREFIXES = (
    "[error",
    "error:",
    "traceback (most recent call last)",
    "internal server error",
)
_ERROR_TEXT_MARKERS = (
    "llamacpp failed",
    "client error '4",
    "client error '5",
    "server error '5",
    "connection refused",
    "read timeout",
)


def looks_like_error_text(text: str) -> bool:
    """True when generated 'prose' is really a plumbing error report.

    Backstop only — the ok/error fields on the result contract are the primary
    gate. This exists because an upstream can report failure purely in the text
    (confirmed live: a llamacpp 400 arrived as a non-empty final_text). Kept
    deliberately narrow: it matches error *framing*, not the mere presence of
    the word "error", so Orion can still say "the codebase is throwing errors
    I can't map yet" -- which it genuinely has.
    """
    stripped = str(text or "").strip().lower()
    if not stripped:
        return False
    if stripped.startswith(_ERROR_TEXT_PREFIXES):
        return True
    # Markers only count near the start; a long reflective passage that happens
    # to mention a timeout deep in the body is not an error report.
    head = stripped[:200]
    return any(marker in head for marker in _ERROR_TEXT_MARKERS)


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
        probability: float,
        min_cooldown_sec: float,
        daily_cap: int,
        quiet_start_hour: int,
        quiet_end_hour: int,
        llm_route: str,
        timeout_sec: float,
        notify_channel: str,
        fallback_session_id: str,
        timezone_name: str = "UTC",
        rng: Optional[random.Random] = None,
    ) -> None:
        self.enabled = enabled
        self.tick_interval_sec = max(5.0, float(tick_interval_sec))
        self.probability = min(1.0, max(0.0, float(probability)))
        self.min_cooldown_sec = max(0.0, float(min_cooldown_sec))
        self.daily_cap = int(daily_cap)
        self.quiet_start_hour = int(quiet_start_hour)
        self.quiet_end_hour = int(quiet_end_hour)
        self.llm_route = str(llm_route or "quick").strip().lower()
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
        self._rng = rng or random.Random()

        self._bus: Any = None
        self._cortex_client: Any = None
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

    async def start(self, bus: Any, cortex_client: Any) -> None:
        self._bus = bus
        self._cortex_client = cortex_client
        if not self.enabled:
            logger.info("endogenous_outreach disabled")
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="hub-endogenous-outreach")
        logger.info(
            "endogenous_outreach started tick=%.0fs p=%.2f cooldown=%.0fs cap=%d quiet=%d-%d route=%s",
            self.tick_interval_sec,
            self.probability,
            self.min_cooldown_sec,
            self.daily_cap,
            self.quiet_start_hour,
            self.quiet_end_hour,
            self.llm_route,
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
            "probability": self.probability,
            "min_cooldown_sec": self.min_cooldown_sec,
            "daily_cap": self.daily_cap,
            "quiet_hours": [self.quiet_start_hour, self.quiet_end_hour],
            "timezone": self.timezone_name,
            "llm_route": self.llm_route,
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

    def _should_roll(self) -> bool:
        """STUB. Replace with a real endogenous trigger when one exists."""
        return self._rng.random() < self.probability

    # -- the tick ----------------------------------------------------------

    async def maybe_outreach(self, *, force: bool = False) -> Dict[str, Any]:
        """One decision cycle. Returns a status dict; never raises.

        ``force`` skips the random roll and NOTHING else — every safety gate
        still applies, including ``enabled``. The debug endpoint that calls this
        is unauthenticated, so a carve-out here would make "off by default" a
        lie that one POST could undo.
        """
        if self._send_lock.locked():
            return self._record({"outreach": False, "reason": "already_sending"})
        async with self._send_lock:
            return await self._outreach_once(force=force)

    async def _outreach_once(self, *, force: bool) -> Dict[str, Any]:
        blocked = outreach_block_reason(self._gate_inputs())
        if blocked:
            return self._record({"outreach": False, "reason": blocked})
        if not force and not self._should_roll():
            return self._record({"outreach": False, "reason": "not_rolled"})

        session_id = self._active_session_id()
        ctx = await self._gather_context(session_id)
        prompt = build_outreach_prompt(ctx)
        if not prompt:
            return self._record({"outreach": False, "reason": "no_grounding_context"})

        correlation_id = str(uuid4())
        raw_text, gen_debug = await self._generate(prompt, session_id, correlation_id)
        # Strip here as well as in _generate: this is where the ship/drop
        # decision is made, so whitespace-only output must not slip past on the
        # assumption that the producer already normalized it.
        text = str(raw_text or "").strip()
        if not text:
            return self._record(
                {"outreach": False, "reason": "empty_generation", "generation": gen_debug}
            )
        if is_pass_response(text):
            return self._record(
                {"outreach": False, "reason": "orion_passed", "generation": gen_debug}
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
                }
            )

        await self._deliver(text=text, session_id=session_id, correlation_id=correlation_id)

        self._last_outreach_at = time.time()
        self._sent_today += 1
        logger.info(
            "endogenous_outreach_sent corr=%s session=%s route=%s chars=%d sent_today=%d",
            correlation_id,
            session_id,
            self.llm_route,
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
            }
        )

    def _record(self, result: Dict[str, Any]) -> Dict[str, Any]:
        result["at"] = datetime.now(timezone.utc).isoformat()
        self._last_result = result
        return result

    async def _gather_context(self, session_id: Optional[str]) -> OutreachContext:
        """Read grounding signals off the main loop; failures degrade to empty."""

        async def _safe(fn, *args):
            try:
                return await asyncio.to_thread(fn, *args)
            except Exception as exc:  # noqa: BLE001
                logger.warning("endogenous_outreach_context_read_failed fn=%s err=%s", fn.__name__, exc)
                return None

        summaries = await _safe(_fetch_curiosity_summaries)
        turns = await _safe(_fetch_recent_turns, session_id)

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
        )

    async def _generate(
        self, prompt: str, session_id: str, correlation_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Quick/metacog-lane cortex call. Returns ("", debug) on any failure."""
        if not self._cortex_client:
            return "", {"error": "no_cortex_client"}

        from orion.schemas.cortex.contracts import CortexChatRequest

        req = CortexChatRequest(
            prompt=prompt,
            mode="brain",
            session_id=session_id,
            options={
                "llm_route": self.llm_route,
                # These three are the keys the executor actually reads. A bare
                # options["no_write"] is inert here: the only thing that
                # translates it is cortex_request_builder (which reads it as a
                # TOP-LEVEL payload key, not an option), and this module calls
                # cortex_client directly rather than going through that builder.
                # orion-cortex-exec/app/supervisor.py reads no_write_active.
                "tool_execution_policy": "none",
                "action_execution_policy": "none",
                "no_write_active": True,
                "source": OUTREACH_TAG,
            },
            # Recall is controlled by the typed `recall` field, not by an
            # option: orion-cortex-gateway/app/bus_client.py builds a
            # RecallDirective from it and falls back to RecallDirective()
            # (enabled=True) when it is None. An unsolicited tick has no user
            # query to retrieve against, so leaving this unset would fire full
            # default recall on every outreach.
            recall={"enabled": False},
            metadata={"source": OUTREACH_TAG, "unsolicited": True},
        )
        started = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                self._cortex_client.chat(req, correlation_id=correlation_id),
                timeout=self.timeout_sec,
            )
        except (TimeoutError, asyncio.TimeoutError):
            return "", {"error": "timeout", "timeout_sec": self.timeout_sec}
        except Exception as exc:  # noqa: BLE001
            logger.warning("endogenous_outreach_generate_failed corr=%s err=%s", correlation_id, exc)
            return "", {"error": type(exc).__name__, "detail": str(exc)[:240]}

        text = str(getattr(resp, "final_text", "") or "").strip()
        elapsed = round(time.monotonic() - started, 3)
        cortex_result = getattr(resp, "cortex_result", None)
        ok = bool(getattr(cortex_result, "ok", False))
        status = str(getattr(cortex_result, "status", "") or "")
        cortex_error = getattr(cortex_result, "error", None)

        # An emptiness check is NOT sufficient. Confirmed live 2026-08-14: a
        # llamacpp 400 came back through this path as a perfectly non-empty
        # final_text ("[Error: llamacpp failed: Client error '400 Bad
        # Request'...]"), sailed past `if not text`, and got delivered and
        # persisted into Juniper's real chat thread as if Orion had said it.
        # That is exactly the "fallback text masquerading as generated
        # cognition" AGENTS.md §0A bans. Gate on the result contract's own
        # ok/error fields, and refuse error-shaped prose as a backstop for
        # upstreams that report failure only in the text.
        if not ok or cortex_error:
            return "", {
                "error": "cortex_not_ok",
                "status": status,
                "ok": ok,
                "cortex_error": str(cortex_error)[:240] if cortex_error else None,
                "elapsed_sec": elapsed,
                "llm_route": self.llm_route,
            }
        if looks_like_error_text(text):
            logger.warning(
                "endogenous_outreach_error_shaped_text corr=%s status=%s text=%r",
                correlation_id,
                status,
                text[:200],
            )
            return "", {
                "error": "error_shaped_text",
                "status": status,
                "elapsed_sec": elapsed,
                "llm_route": self.llm_route,
            }

        debug = {
            "elapsed_sec": elapsed,
            "llm_route": self.llm_route,
            "final_len": len(text),
            "status": status,
        }
        return text, debug

    # -- delivery ----------------------------------------------------------

    async def _deliver(self, *, text: str, session_id: str, correlation_id: str) -> None:
        message_id = str(uuid4())
        self._push_to_sockets(text=text, session_id=session_id, correlation_id=correlation_id, message_id=message_id)
        await self._publish_history(
            text=text, session_id=session_id, correlation_id=correlation_id, message_id=message_id
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
            "llm_route": self.llm_route,
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
        self, *, text: str, session_id: str, correlation_id: str, message_id: str
    ) -> None:
        try:
            from scripts.chat_history import build_chat_history_envelope, publish_chat_history

            env = build_chat_history_envelope(
                content=text,
                role="assistant",
                session_id=session_id,
                correlation_id=correlation_id,
                speaker="Orion",
                tags=[OUTREACH_TAG],
                message_id=message_id,
                client_meta={"unsolicited": True, "llm_route": self.llm_route},
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
