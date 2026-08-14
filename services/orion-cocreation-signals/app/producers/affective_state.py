"""affective_state producer: polls Juniper's real local Claude Code transcripts
on an interval, publishes a ``JuniperAffectiveStateV1`` event every tick --
same "a time window always has a real answer, even an all-zero one" reasoning
as ``pr_lifecycle_loop`` (docs/superpowers/specs/2026-07-30-juniper-affective-
state-signal-proposal.md).

Half-open ``[window_since, window_until)`` windows tile contiguously tick-to-
tick, same convention as ``pr_lifecycle_loop``. Same real, acknowledged
restart-loss gap too: this is a time-window scan, not a diff between two
saved states, so a window entirely inside a downtime longer than
``cold_start_lookback_sec`` is genuinely lost, not recovered by a bigger scan
next time.

Scans every transcript file under ``claude_projects_path`` on every tick,
filtered to messages whose timestamp falls in the current window -- not an
incremental per-file offset cursor. Accepted simplification for this first
cut (real corpus at proposal time: ~1200 files, sub-second full walk); if
this ever becomes a real bottleneck, revisit with per-file byte offsets
rather than optimizing before there's a measured problem to fix.

Publishes only the aggregate scalar (``swear_frequency``) -- never persists
or logs raw message text at any point in this loop, per the proposal doc's
privacy boundary. ``typo_rate`` is deliberately not computed here at all
(not just omitted from the wire schema) -- see
``orion/schemas/affective_state.py``'s docstring for why.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.cocreation.affective_signals import aggregate_scores, score_message
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.dev_economics.claude_code_ingest import iter_all_human_messages
from orion.schemas.affective_state import JuniperAffectiveStateV1

logger = logging.getLogger("orion.cocreation_signals.affective_state")


def _score_window(
    claude_projects_path: str, since: datetime, until: datetime, *, cold_start: bool = False
) -> JuniperAffectiveStateV1:
    """Real, synchronous scan+score -- caller runs this via asyncio.to_thread
    (parsing+spellchecking a real transcript tree is blocking I/O/CPU work,
    same reasoning as git_delta_loop wrapping its git subprocess calls)."""
    messages_in_window = [
        m for m in iter_all_human_messages(claude_projects_path) if since <= m.timestamp < until
    ]
    # compute_typos=False: this schema never exposes typo_rate (see
    # orion/schemas/affective_state.py's own docstring for why), so running
    # a real spellcheck pass over every word in every message here would be
    # pure waste -- confirmed live 2026-08-11 (code review) against a real
    # tick that spellchecked ~31k words for a value nothing downstream read.
    agg = aggregate_scores(
        [score_message(m.text, compute_typos=False) for m in messages_in_window]
    )
    return JuniperAffectiveStateV1(
        observed_at=datetime.now(timezone.utc),
        window_since=since,
        window_until=until,
        message_count=len(messages_in_window),
        word_count=agg.word_count,
        swear_count=agg.swear_count,
        swear_frequency=agg.swear_frequency,
        cold_start=cold_start,
    )


async def _publish(bus: OrionBusAsync, channel: str, source: ServiceRef, event) -> bool:
    """Same True/False publish contract as git_delta_loop/pr_lifecycle_loop's
    own ``_publish()`` -- False only on a real transient publish failure, so
    the caller knows not to advance its window cursor."""
    if not getattr(bus, "enabled", False):
        logger.info("cocreation_affective_state_publish_skipped_bus_disabled")
        return True
    envelope = BaseEnvelope(
        kind="juniper.affective_state.v1", source=source, payload=event.model_dump(mode="json")
    )
    try:
        await bus.publish(channel, envelope)
        logger.info(
            "cocreation_affective_state_published message_count=%d word_count=%d "
            "swear_count=%d swear_frequency=%s",
            event.message_count, event.word_count, event.swear_count, event.swear_frequency,
        )
        return True
    except Exception:
        logger.exception("cocreation_affective_state_publish_failed")
        return False


async def affective_state_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    claude_projects_path: str,
    poll_interval_sec: float,
    cold_start_lookback_sec: float,
    stop: asyncio.Event,
) -> None:
    if not Path(claude_projects_path).exists():
        # Fail loud once at startup rather than publishing an empty window
        # every tick forever, which would misreport "checked, found nothing"
        # instead of the real "mount is missing/misconfigured" fact.
        logger.error(
            "cocreation_affective_state_claude_projects_path_missing path=%s -- "
            "producer will not start; check COCREATION_SIGNALS_CLAUDE_PROJECTS_HOST_PATH",
            claude_projects_path,
        )
        return

    last_until = datetime.now(timezone.utc) - timedelta(seconds=cold_start_lookback_sec)
    is_cold_start = True
    while not stop.is_set():
        try:
            until = datetime.now(timezone.utc)
            # `cold_start` marks the FIRST tick after a (re)start, whose window
            # is cold_start_lookback_sec wide rather than one poll interval --
            # 3600s against a 900s poll in the live config, so it re-covers up
            # to four windows a previous run already published.
            #
            # Ordinary ticks tile exactly (this tick's window_since is the last
            # one's window_until), so they can be summed. A cold-start window
            # cannot be summed with them without double-counting: confirmed on
            # the first two real rows ever persisted, where a 913s window sat
            # entirely inside a 3600s one and summing inflated the hour by
            # +34.7%. Flagging it lets a consumer sum the tiling rows and use
            # cold-start rows only to fill genuine downtime gaps.
            event = await asyncio.to_thread(
                _score_window, claude_projects_path, last_until, until, cold_start=is_cold_start
            )
            published = await _publish(bus, channel, source, event)
            if published:
                last_until = until
                # Cleared only on a SUCCESSFUL publish, for the same reason
                # last_until is: a failed publish leaves the cursor parked, so
                # the retry covers an even wider range that still overlaps rows
                # a previous run published. Clearing the flag unconditionally
                # would hand that retry out as an ordinary tiling window.
                is_cold_start = False
            # else: leave last_until unchanged so the next tick's window
            # extends to cover this failed range too, instead of silently
            # dropping it (same reasoning as pr_lifecycle_loop).
        except Exception:
            logger.exception("cocreation_affective_state_tick_failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval_sec)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
