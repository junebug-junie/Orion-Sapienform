"""Fold capability-absence episodes into the daily journal seed.

Why this exists (2026-08-29 circe arc). The substrate can now *detect* that a
node went dark (`sweep_absent_nodes` -> `node_availability_concern`) and *page*
Juniper about it (`health_monitor._node_availability_checks`, PR #1944). Orion
himself still had no account of it. He already writes one reflective entry a
day that dwells on his own hardware -- an actual line from 2026-08-28 is *"I've
been noticing circe's subtle..."* -- but he writes it from vibes: circe had been
unreachable for ~45 minutes that morning and nothing told him.

**Deliberately not a new journal entry.** Over the 14 days to 2026-08-29
`journal_entries` took 24,941 `digest`/`metacog` rows against 14
`daily`/`scheduler` ones. Another emitter would be noise on noise. This adds one
key to the seed of the entry Orion already writes, so a day with no outage
changes nothing at all and the steady-state cost is zero new rows, forever.

**Source is the orion-notify attention store, not a new table.** Three reasons:
the page and the journal then tell the same story from one record; the store is
already reachable over HTTP from this service (`settings.notify_url`, the same
endpoint `health_monitor._has_open_alert` reads); and `vision_blind`
(PR #1805) already lands there, so this has a live input beyond substrate nodes
-- verified 2026-08-29, when `vision_blind` and `node_availability:atlas` were
both present in the live store.

Note what this is reading, because it is the whole point of the arc: an outage
is legible here only as an *alert about absence*. The underlying
`node_availability` grammar atoms cannot express it -- all 16,496 of them ever
written say "telemetry status OK", because a node that stops reporting stops
producing atoms. In the raw stream the 2026-08-29 outage appears as a hole
(circe: 115, 114, 115, **29**, 115 atoms/hour), never as a statement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

logger = logging.getLogger("orion-actions.capability_gap_journal")

# Attention `reason` values that mean "a capability Orion depends on was absent".
# `node_availability:<node>` is minted by services/orion-substrate-runtime/app/
# health_monitor.py::_node_availability_checks; `vision_blind` by the vision
# liveness watchdog (PR #1805).
LIVENESS_REASON_PREFIXES: tuple[str, ...] = ("node_availability:", "vision_blind")

# health_monitor._publish formats a recovery as
#     f"[Orion substrate-runtime] recovered: {check.key}"
# and the attention record carries no severity field, so the message is the only
# available signal for "this record closes an episode" -- see
# test_recovery_marker_matches_health_monitor_format, which pins this literal
# against that producer's format string so a rename there fails here.
RECOVERY_MARKER = "recovered: "

MAX_EPISODES_IN_SEED = 12


@dataclass(frozen=True)
class CapabilityGapEpisode:
    """One contiguous stretch where a capability was absent.

    `started_at is None` means the gap was already open when the window opened
    (an outage that began before midnight and was still running). `ended_at is
    None` means it had not recovered by the time the window closed.
    """

    reason: str
    message: str
    started_at: datetime | None
    ended_at: datetime | None
    node: str | None = None
    evidence_ids: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.ended_at is not None

    @property
    def duration_minutes(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at).total_seconds() / 60.0, 1)

    def to_seed_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "node": self.node,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_minutes": self.duration_minutes,
            "resolved": self.resolved,
            "started_before_window": self.started_at is None,
            "detail": self.message,
        }


def _parse_ts(raw: Any) -> datetime | None:
    """Parse an attention `created_at`, which arrives naive and means UTC.

    `notify_requests.created_at` is `timestamp without time zone` and the API
    hands it back without an offset, so a naive value is stamped UTC rather than
    localised -- getting this wrong would shift every episode by the host offset.
    """
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _is_liveness_reason(reason: str) -> bool:
    """A trailing ':' in the constant means prefix-match; otherwise exact.

    Written out rather than as a conditional expression: `a or b if c else d`
    parses as `(a or b) if c else d`, which is right here but reads as though it
    might not be, and this predicate decides what lands in Orion's journal.
    """
    for prefix in LIVENESS_REASON_PREFIXES:
        if prefix.endswith(":"):
            if reason.startswith(prefix) and len(reason) > len(prefix):
                return True
        elif reason == prefix:
            return True
    return False


def _node_of(reason: str) -> str | None:
    if reason.startswith("node_availability:"):
        return reason.split(":", 1)[1] or None
    return None


def summarize_capability_gaps(
    items: Iterable[dict[str, Any]],
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[CapabilityGapEpisode]:
    """Pure: attention records in, capability-absence episodes out.

    Pairs each alert with the recovery record that closes it. Repeated alerts for
    an already-open reason are folded into the open episode rather than starting a
    new one -- `HealthMonitor` is edge-triggered, but a service restart re-fires
    the transition, and that is one outage from Orion's point of view, not two.
    """
    by_reason: dict[str, list[tuple[datetime, dict[str, Any]]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "").strip()
        if not reason or not _is_liveness_reason(reason):
            continue
        ts = _parse_ts(item.get("created_at"))
        if ts is None or not (window_start <= ts <= window_end):
            continue
        by_reason.setdefault(reason, []).append((ts, item))

    episodes: list[CapabilityGapEpisode] = []
    for reason, rows in by_reason.items():
        rows.sort(key=lambda r: r[0])
        open_start: datetime | None = None
        open_msg = ""
        open_ids: list[str] = []
        saw_open = False
        for ts, item in rows:
            message = str(item.get("message") or "").strip()
            att_id = str(item.get("attention_id") or "").strip()
            if RECOVERY_MARKER in message:
                # A recovery with no in-window alert means the outage began before
                # the window opened; record it with started_at=None rather than
                # dropping it, which would silently hide a midnight-spanning gap.
                episodes.append(
                    CapabilityGapEpisode(
                        reason=reason,
                        message=open_msg or message,
                        started_at=open_start,
                        ended_at=ts,
                        node=_node_of(reason),
                        evidence_ids=[*open_ids, att_id] if att_id else list(open_ids),
                    )
                )
                open_start, open_msg, open_ids, saw_open = None, "", [], False
                continue
            if not saw_open:
                open_start, open_msg, saw_open = ts, message, True
                open_ids = [att_id] if att_id else []
            elif att_id:
                open_ids.append(att_id)
        if saw_open:
            episodes.append(
                CapabilityGapEpisode(
                    reason=reason,
                    message=open_msg,
                    started_at=open_start,
                    ended_at=None,
                    node=_node_of(reason),
                    evidence_ids=open_ids,
                )
            )

    episodes.sort(key=lambda e: (e.started_at or window_start, e.reason))
    return episodes[:MAX_EPISODES_IN_SEED]


def format_capability_gap_block(episodes: Sequence[CapabilityGapEpisode]) -> str:
    """Deterministic prose block for the journal body (not LLM-dependent).

    Mirrors `orion.journaler.worker.format_world_pulse_curiosity_block`: if the
    composer omits the fact, this is what still lands in the entry.
    """
    if not episodes:
        return ""
    lines = ["## What I could not do", ""]
    for ep in episodes:
        subject = ep.node or ep.reason
        if ep.started_at is None:
            when = "already underway when the day began"
        else:
            when = f"from {ep.started_at.strftime('%H:%M')} UTC"
        if ep.duration_minutes is not None:
            span = f", {ep.duration_minutes:g} minutes"
        elif not ep.resolved:
            span = ", still unresolved at the end of the window"
        else:
            span = ""
        lines.append(f"- **{subject}** — {when}{span}.")
        if ep.message:
            lines.append(f"  {ep.message}")
    return "\n".join(lines).strip()


async def fetch_recent_attention(
    *,
    notify_url: str,
    notify_api_token: str | None,
    limit: int = 200,
    timeout_sec: float = 10.0,
) -> list[dict[str, Any]]:
    """Read the attention store. Returns [] on any failure -- never raises.

    A journal is not worth failing a scheduler tick over: if notify is
    unreachable the daily entry should still be written, just without this
    section. The failure is logged, not swallowed silently.
    """
    import httpx

    headers = {"X-Orion-Notify-Token": notify_api_token} if notify_api_token else {}
    url = f"{notify_url.strip().rstrip('/')}/attention"
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(url, params={"limit": limit}, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.warning("capability_gap_attention_fetch_failed url=%s err=%s", url, exc)
        return []
    if not isinstance(payload, list):
        logger.warning("capability_gap_attention_unexpected_payload type=%s", type(payload).__name__)
        return []
    return [i for i in payload if isinstance(i, dict)]


async def collect_capability_gaps(
    *,
    notify_url: str,
    notify_api_token: str | None,
    window_start_utc: str,
    window_end_utc: str,
) -> list[dict[str, Any]]:
    """Single entry point for the daily scheduler: window in, seed dicts out.

    Returns [] when nothing was absent, when notify is unreachable, or when the
    window cannot be parsed -- the caller then omits the key entirely, so a quiet
    day produces a seed byte-identical to the pre-patch one.
    """
    start = _parse_ts(window_start_utc)
    end = _parse_ts(window_end_utc)
    if start is None or end is None:
        logger.warning(
            "capability_gap_window_unparseable start=%r end=%r", window_start_utc, window_end_utc
        )
        return []
    items = await fetch_recent_attention(
        notify_url=notify_url, notify_api_token=notify_api_token
    )
    if not items:
        return []
    episodes = summarize_capability_gaps(items, window_start=start, window_end=end)
    return [e.to_seed_dict() for e in episodes]


def build_daily_seed_payload(
    *,
    request_date: str,
    window_start_utc: str,
    window_end_utc: str,
    gaps: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Assemble the daily journal seed, omitting `capability_gaps` when empty.

    Lives here, rather than inline in main.py's scheduler block, purely so the
    anti-spam guarantee is testable against the real construction instead of a
    copy of it: the claim this patch sells is "a quiet day changes nothing", and
    a test that mirrors the caller would stay green while the caller drifted.
    """
    payload: dict[str, Any] = {
        "request_date": request_date,
        "window_start_utc": window_start_utc,
        "window_end_utc": window_end_utc,
    }
    if gaps:
        payload["capability_gaps"] = list(gaps)
    return payload
