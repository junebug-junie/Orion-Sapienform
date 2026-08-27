"""dev_economics producer: scans Juniper's real local Claude Code transcripts
on an interval, publishes a ``DevEconomicsLedgerV1`` event of the real
*growth* in token/word/cost totals since the last check --
docs/superpowers/specs/2026-07-30-dev-economics-signal-design.md.

Deliberately NOT window-membership filtering by ``started_at`` (the first
draft's approach, and ``affective_state_loop``'s own pattern) -- code review
2026-08-12 caught that failing for the common case here: a
``SessionUsageRecord`` is a *cumulative snapshot of the whole transcript
file*, not a discrete per-message event like ``affective_state_loop``
scores. Filtering by ``started_at`` alone means any session that outlives
one poll tick (the normal case at a 900s cadence) gets counted once, on
whichever tick first observed its start, and then silently drops out of
every later tick forever -- its real ongoing growth never gets attributed to
any window at all.

Instead this producer tracks each transcript file's last-observed cumulative
totals in-process (keyed by ``transcript_path``) and publishes the real
**delta** since the last tick -- same "diff since last known state" shape as
``git_delta_loop``, just per-transcript-file instead of per-repo-HEAD. The very
first tick is a true cold start (seeds the baseline, publishes nothing) --
same convention as ``git_delta_loop``'s own cold start, and a deliberate
departure from ``affective_state_loop``'s "publish every tick, even
all-zero" contract, since there is no real "zero delta" fact to report
before a baseline exists.

Same real, acknowledged restart-loss gap as every other producer here: the
in-process ``last_totals`` state is lost on a container restart, so activity
between the last successful tick and the restart is folded into the next
cold-start baseline rather than reported as a delta -- accepted, same shape
as ``git_delta_loop``'s own restart-loss story (the next tick's baseline
just starts a bit later than it "should").

Shares the same ``claude_projects_path`` mount as ``affective_state_loop``
(the whole real ``~/.claude/projects`` tree). Publishes only real,
already-aggregated token/word/cost counts -- never persists or logs raw
message text at any point in this loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.dev_economics.claude_code_ingest import SessionUsageRecord, iter_all_session_usage_records
from orion.dev_economics.ledger_aggregate import (
    aggregate_session_deltas,
    diff_session_record,
    has_real_delta,
)
from orion.schemas.dev_economics import DevEconomicsLedgerV1

logger = logging.getLogger("orion.cocreation_signals.dev_economics")


def _scan_totals(claude_projects_path: str) -> dict[Path, SessionUsageRecord]:
    """Real, synchronous full-tree scan -- caller runs this via
    asyncio.to_thread, same reasoning as every other producer's own I/O
    boundary.

    Keyed by the record's *resolved* ``transcript_path``, which is the only
    key that is actually one-per-record. ``session_id`` is NOT unique: a
    session and every subagent it dispatches share the parent's
    ``session_id`` while each getting its own transcript file and its own
    ``SessionUsageRecord`` (deliberately -- see that class's docstring on
    why subagent usage must not fold into its parent's). Keying this dict by
    ``session_id`` therefore silently kept only the last record per session
    and dropped the rest, which is exactly the systematic undercount
    ``SessionUsageRecord``'s docstring warns about.

    Measured 2026-08-27 against the real transcript tree: 1250 records
    collapsed to 93 keys -- 1157 records (92.6%) discarded, raising visible
    cumulative tokens 15.21B -> 18.74B once fixed (a 1.23x correction), with
    one ``session_id`` owning 117 files. Those absolute counts are a live,
    growing, Claude-Code-pruned tree and will not reproduce exactly; the
    collapse and the ratio are the reproducible facts. Re-measure as
    ``len(recs)`` vs ``len({r.session_id for r in recs})`` over
    ``iter_all_session_usage_records``.

    Downstream never saw it, because ``diff_session_record``'s
    ``max(0, ...)`` truncation guard launders the resulting negative into a
    clean zero, and because ``aggregate_session_deltas`` splits on
    ``is_subagent`` -- so the dropped records surfaced only as
    ``subagent_transcript_count`` sitting at exactly 0 across every live
    tick for the preceding two weeks.

    ``.resolve()`` matters: Claude Code writes some cross-project subagent
    transcripts as absolute-path symlinks back into this same root (see
    ``claude_code_ingest``'s module docstring and this service's
    docker-compose comment), and ``os.walk`` yields a symlink-to-file in
    ``filenames``, so one underlying file can be walked twice under two
    spellings. ``session_id`` keying happened to collapse those; raw-path
    keying would double-count them. This also covers hardlinks."""
    return {
        r.transcript_path.resolve(): r
        for r in iter_all_session_usage_records(claude_projects_path)
    }


def _predates_baseline(
    record: SessionUsageRecord, baseline_taken_at: datetime | None
) -> bool:
    """True when ``record`` is a transcript that already existed before the
    baseline scan was taken -- i.e. seeing it for the first time now means
    it was re-discovered, not created. See ``_score_tick``'s docstring."""
    if baseline_taken_at is None or record.started_at is None:
        return False
    return record.started_at < baseline_taken_at


def _score_tick(
    claude_projects_path: str,
    last_totals: dict[Path, SessionUsageRecord],
    observed_at: datetime,
    baseline_taken_at: datetime | None = None,
) -> tuple[DevEconomicsLedgerV1, dict[Path, SessionUsageRecord]]:
    """One tick's real delta against ``last_totals`` -- returns the event to
    publish and the new totals snapshot to carry into the next tick.
    Synchronous; run via asyncio.to_thread by the caller.

    ``baseline_taken_at`` is when ``last_totals`` was scanned. It exists to
    contain one real hazard: ``diff_session_record(previous=None, current)``
    returns the file's *entire cumulative history* as this tick's growth,
    which is correct for a genuinely new transcript but catastrophically
    wrong for a **re-discovered** one. Two live mechanisms can transiently
    hide a file and then hand it back: ``iter_transcript_files`` walks with
    ``os.walk(..., onerror=lambda _exc: None)``, which silently drops an
    entire subtree on any transient scandir error (EACCES/ESTALE on the
    read-only bind mount), and ``iter_all_session_usage_records`` skips
    unresolvable absolute symlinks per-file -- so the moment an operator
    fixes the remapped mount this service's own docker-compose comment tells
    them to fix, every previously-skipped transcript reappears at once.
    Either path would republish accumulated history as a single tick's
    spend: the largest single session subtree measured on the real tree is
    1.28B tokens, and ``dev_economics_ledger_log.total_tokens`` is a
    Postgres ``integer`` (ceiling 2.15B), so the outcome is a fabricated
    spend spike or an outright ``NumericValueOutOfRange`` on that message.

    So a first-seen path whose transcript demonstrably *started before the
    previous scan* is treated as a baseline to seed, not as growth: it is
    recorded into the returned totals and contributes nothing to this
    event. A first-seen path that started after the previous scan is
    genuinely new work and is published in full, which is the whole reason
    the ``previous=None`` branch exists. A record with no parseable
    ``started_at`` is treated as new -- such a transcript has no usable
    timestamps and in practice carries no usage either, so publishing it
    cannot manufacture a spike, whereas suppressing it could silently drop
    real spend.

    When ``baseline_taken_at`` is None the guard is inert and behaviour is
    the historical one."""
    current_totals = _scan_totals(claude_projects_path)

    deltas = []
    seeded = 0
    for transcript_path, record in current_totals.items():
        previous = last_totals.get(transcript_path)
        if previous is None and _predates_baseline(record, baseline_taken_at):
            seeded += 1
            continue
        deltas.append(diff_session_record(previous, record))
    if seeded:
        logger.info(
            "cocreation_dev_economics_seeded_rediscovered_transcripts count=%d "
            "-- pre-existing history withheld from this tick's delta",
            seeded,
        )
    real_deltas = [d for d in deltas if has_real_delta(d)]
    agg = aggregate_session_deltas(real_deltas, observed_at=observed_at)
    event = DevEconomicsLedgerV1(
        observed_at=observed_at,
        window_since=observed_at,  # bookkeeping only now -- see module docstring
        window_until=observed_at,
        session_count=agg.session_count,
        subagent_transcript_count=agg.subagent_transcript_count,
        total_input_tokens=agg.total_input_tokens,
        total_output_tokens=agg.total_output_tokens,
        total_cache_creation_input_tokens=agg.total_cache_creation_input_tokens,
        total_cache_read_input_tokens=agg.total_cache_read_input_tokens,
        total_tokens=agg.total_tokens,
        total_assistant_word_count=agg.total_assistant_word_count,
        total_human_word_count=agg.total_human_word_count,
        total_estimated_cost_usd=agg.total_estimated_cost_usd,
        priced_session_count=agg.priced_session_count,
        unpriced_session_count=agg.unpriced_session_count,
        model_mix=agg.model_mix,
    )
    return event, current_totals


async def _publish(bus: OrionBusAsync, channel: str, source: ServiceRef, event) -> bool:
    """Same True/False publish contract as the other producers' own
    ``_publish()`` -- False only on a real transient publish failure, so the
    caller knows not to advance its ``last_totals`` baseline past this
    delta (otherwise the growth this tick observed would be silently lost
    instead of folded into the next tick's bigger delta)."""
    if not getattr(bus, "enabled", False):
        logger.info("cocreation_dev_economics_publish_skipped_bus_disabled")
        return True
    envelope = BaseEnvelope(
        kind="substrate.dev_economics_ledger.v1", source=source, payload=event.model_dump(mode="json")
    )
    try:
        await bus.publish(channel, envelope)
        logger.info(
            "cocreation_dev_economics_published session_count=%d total_tokens=%d "
            "total_estimated_cost_usd=%s priced_session_count=%d unpriced_session_count=%d",
            event.session_count, event.total_tokens, event.total_estimated_cost_usd,
            event.priced_session_count, event.unpriced_session_count,
        )
        return True
    except Exception:
        logger.exception("cocreation_dev_economics_publish_failed")
        return False


async def dev_economics_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    claude_projects_path: str,
    poll_interval_sec: float,
    stop: asyncio.Event,
) -> None:
    if not Path(claude_projects_path).exists():
        # Fail loud once at startup rather than publishing an empty/no-op
        # tick forever, which would misreport "checked, found nothing"
        # instead of the real "mount is missing/misconfigured" fact.
        logger.error(
            "cocreation_dev_economics_claude_projects_path_missing path=%s -- "
            "producer will not start; check COCREATION_SIGNALS_CLAUDE_PROJECTS_HOST_PATH",
            claude_projects_path,
        )
        return

    last_totals: dict[Path, SessionUsageRecord] | None = None
    # When last_totals was scanned. Advances in lockstep with it, so a failed
    # publish keeps both the totals and their timestamp, and the next tick's
    # delta still covers the whole range since the last *accepted* baseline.
    baseline_taken_at: datetime | None = None
    while not stop.is_set():
        try:
            observed_at = datetime.now(timezone.utc)
            if last_totals is None:
                last_totals = await asyncio.to_thread(_scan_totals, claude_projects_path)
                baseline_taken_at = observed_at
                logger.info(
                    "cocreation_dev_economics_cold_start transcript_count=%d", len(last_totals)
                )
            else:
                event, current_totals = await asyncio.to_thread(
                    _score_tick,
                    claude_projects_path,
                    last_totals,
                    observed_at,
                    baseline_taken_at,
                )
                published = await _publish(bus, channel, source, event)
                if published:
                    last_totals = current_totals
                    baseline_taken_at = observed_at
                # else: leave last_totals unchanged -- the next tick's delta
                # will naturally cover this failed range too, instead of
                # silently losing it (same reasoning as git_delta_loop's own
                # _publish() contract).
        except Exception:
            logger.exception("cocreation_dev_economics_tick_failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval_sec)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
