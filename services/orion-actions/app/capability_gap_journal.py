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

# Alert `reason` -> the `reason` its producer uses to announce recovery.
# A producer that announces recovery under the SAME reason (health_monitor does:
# `_publish` reuses `check.key` for both) maps to itself.
#   node_availability:<node>  services/orion-substrate-runtime/app/health_monitor.py (PR #1944)
#   vision_blind              services/orion-vision-host/app/liveness.py (PR #1805)
RECOVERY_REASON_BY_ALERT: dict[str, str] = {"vision_blind": "vision_recovered"}

# Alert reasons this module tracks. A trailing ':' means prefix-match.
LIVENESS_REASON_PREFIXES: tuple[str, ...] = ("node_availability:", "vision_blind")

# Recovery records are identified by `severity == "info"` (health_monitor._publish
# sets exactly that, versus "critical" on the alert; vision-host sets it on
# `vision_recovered`), with this message marker as a secondary signal for the
# same-reason producers.
#
# An earlier version of this module used the marker ALONE and justified it in a
# docstring claiming the attention record "carries no severity field". That was
# false -- `severity` is declared on ChatAttentionState (orion/schemas/notify.py)
# and is present in the live payload; the claim came from reading a truncated
# key listing. Severity is the better signal and is now primary. Kept as a
# belt-and-braces check because a message-only match is exploitable: an alert
# whose interpolated capability list happened to contain "recovered: " would
# otherwise be journaled as a gap that ENDED, at the moment a node went down.
RECOVERY_MARKER = "recovered: "
RECOVERY_SEVERITIES = frozenset({"info"})

# Cap on episodes carried into the seed. Selection keeps the most *recent*
# episodes and unresolved ones first -- an ongoing outage matters more to the
# entry than an old closed one. Display order is chronological.
MAX_EPISODES_IN_SEED = 12

# Attention messages are operator-facing runbook text written by other services
# (the live vision_blind message is ~330 chars of docker/VRAM instructions).
# Truncated before it reaches a prompt: unbounded producer text has no business
# being interpolated verbatim, and a future producer could put a path or token
# in there.
MAX_DETAIL_CHARS = 320


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
    # True when `ended_at` was inferred from a LATER alert rather than read from a
    # recovery record.
    #
    # CORRECTED 2026-08-30. This originally claimed a later alert *proves* the
    # earlier gap closed, on the reasoning that vision-host must clear
    # `_alerting` before it can re-arm. Root-causing the missing
    # `vision_recovered` records showed that is wrong: the watcher's arm state
    # was in-memory, so a restart re-armed it without any recovery having
    # happened. A later alert therefore proves only that the watcher STARTED
    # OVER -- which may be a genuine recovery or may be a restart mid-outage.
    #
    # The bound is still the right behaviour (without it every vision episode
    # since 2026-08-21 stayed permanently open -- `vision_recovered` has never
    # once been emitted, 0 rows ever -- and a 24h window inherited all nine),
    # but it is an upper bound on an *unknown* end, not a measured one. Hence
    # `resolved: false` and `duration_upper_bound_minutes` rather than a hard
    # duration. Once the vision-host arm state is durable, a re-arm becomes
    # meaningful again and this stays correct either way.
    end_is_upper_bound: bool = False

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
            "duration_minutes": None if self.end_is_upper_bound else self.duration_minutes,
            "duration_upper_bound_minutes": self.duration_minutes if self.end_is_upper_bound else None,
            "resolved": self.resolved and not self.end_is_upper_bound,
            "ended_by_a_later_alert": self.end_is_upper_bound,
            "started_before_window": self.started_at is None,
            "detail": _truncate(self.message),
        }


def _truncate(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


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
    """A trailing ':' in the constant means prefix-match; otherwise exact."""
    for prefix in LIVENESS_REASON_PREFIXES:
        if prefix.endswith(":"):
            if reason.startswith(prefix) and len(reason) > len(prefix):
                return True
        elif reason == prefix:
            return True
    return False


def _episode_key(reason: str) -> str | None:
    """The alert reason an incoming record belongs to, or None if irrelevant.

    A recovery record announced under a different reason (`vision_recovered`)
    is mapped back onto the alert it closes; without this it was discarded by
    the liveness filter before it could ever close anything, which is how three
    separate vision outages on 2026-08-29 rendered as one 1h48m span that never
    ended.
    """
    if _is_liveness_reason(reason):
        return reason
    for alert, recovery in RECOVERY_REASON_BY_ALERT.items():
        if reason == recovery:
            return alert
    return None


def _is_recovery(item: dict[str, Any], *, reason: str, episode_key: str) -> bool:
    if reason != episode_key:
        return True  # arrived under a paired recovery reason
    severity = str(item.get("severity") or "").strip().lower()
    if severity in RECOVERY_SEVERITIES:
        return True
    return RECOVERY_MARKER in str(item.get("message") or "") and severity == ""


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

    Episodes are built from the FULL record history and then filtered by
    *overlap* with the window, not by whether individual records fall inside it.
    Filtering records instead means an outage that began before the window and
    was still running at the end of it has neither an in-window alert nor an
    in-window recovery, and vanishes -- day two of a multi-day outage would be
    silent, which is precisely the silence this module exists to remove.

    Repeat alerts are NOT folded into one episode. Folding was in an earlier
    version to absorb a restart re-firing an edge-triggered transition, but
    `health_monitor._has_open_alert` already suppresses that: the 2026-08-29
    absence sweep fired 142 times and produced exactly one attention record. For
    vision-host, which re-arms after clearing, folding actively lied -- three
    distinct outages became one. An alert opening while another is still open
    leaves the earlier one unresolved, which is honest: with no recovery record
    we do not know when it ended.
    """
    tracked: dict[str, list[tuple[datetime, str, dict[str, Any]]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "").strip()
        if not reason:
            continue
        key = _episode_key(reason)
        if key is None:
            continue
        ts = _parse_ts(item.get("created_at"))
        if ts is None or ts > window_end:
            continue
        tracked.setdefault(key, []).append((ts, reason, item))

    episodes: list[CapabilityGapEpisode] = []
    for key, rows in tracked.items():
        rows.sort(key=lambda r: r[0])
        open_ep: dict[str, Any] | None = None
        for ts, reason, item in rows:
            if _is_recovery(item, reason=reason, episode_key=key):
                if open_ep is not None:
                    episodes.append(
                        CapabilityGapEpisode(
                            reason=key,
                            message=open_ep["message"],
                            started_at=open_ep["started_at"],
                            ended_at=ts,
                            node=_node_of(key),
                            evidence_ids=open_ep["ids"],
                        )
                    )
                    open_ep = None
                elif ts >= window_start:
                    # Recovery with no known open alert: the gap began before any
                    # record we hold. Recorded with started_at=None rather than
                    # dropped, so a window-spanning outage is not silent.
                    episodes.append(
                        CapabilityGapEpisode(
                            reason=key,
                            message=str(item.get("message") or "").strip(),
                            started_at=None,
                            ended_at=ts,
                            node=_node_of(key),
                            evidence_ids=[str(item.get("attention_id") or "")],
                        )
                    )
                continue
            if open_ep is not None:
                # A new alert while one is open means the producer's watcher
                # started over -- genuine recovery, or a restart that wiped its
                # arm state. Either way the earlier gap stops accruing here
                # rather than staying open forever; see `end_is_upper_bound`.
                episodes.append(
                    CapabilityGapEpisode(
                        reason=key,
                        message=open_ep["message"],
                        started_at=open_ep["started_at"],
                        ended_at=ts,
                        node=_node_of(key),
                        evidence_ids=open_ep["ids"],
                        end_is_upper_bound=True,
                    )
                )
            att_id = str(item.get("attention_id") or "").strip()
            open_ep = {
                "started_at": ts,
                "message": str(item.get("message") or "").strip(),
                "ids": [att_id] if att_id else [],
            }
        if open_ep is not None:
            episodes.append(
                CapabilityGapEpisode(
                    reason=key,
                    message=open_ep["message"],
                    started_at=open_ep["started_at"],
                    ended_at=None,
                    node=_node_of(key),
                    evidence_ids=open_ep["ids"],
                )
            )

    overlapping = [e for e in episodes if _overlaps(e, window_start, window_end)]
    # Select the most important MAX_EPISODES_IN_SEED (unresolved first, then most
    # recent), then restore chronological order for display. Slicing a
    # chronologically-sorted list instead would drop the NEWEST episode -- the one
    # most likely to still be happening.
    overlapping.sort(key=lambda e: (e.resolved, -(e.started_at or window_start).timestamp()))
    kept = overlapping[:MAX_EPISODES_IN_SEED]
    kept.sort(key=lambda e: (e.started_at or window_start, e.reason))
    return kept


def _overlaps(ep: CapabilityGapEpisode, window_start: datetime, window_end: datetime) -> bool:
    """True when the gap was in effect at any point inside the window."""
    start = ep.started_at or datetime.min.replace(tzinfo=timezone.utc)
    end = ep.ended_at or datetime.max.replace(tzinfo=timezone.utc)
    return start <= window_end and end >= window_start


async def fetch_recent_attention(
    *,
    notify_url: str,
    notify_api_token: str | None,
    limit: int = 200,
    timeout_sec: float = 10.0,
) -> list[dict[str, Any]]:
    """Read the attention store. Returns [] on any failure -- never raises.

    Uses `requests` on a worker thread rather than httpx: httpx is NOT installed
    in the orion-actions image (`requirements.txt` has requests, fastapi,
    uvicorn, pydantic, redis, PyYAML -- confirmed live with
    `docker exec orion-athena-actions python -c "import httpx"`), and the module
    import sat outside the try, so the first daily tick after deploy would have
    raised ModuleNotFoundError straight through this function into the
    scheduler's shared handler -- taking out the journal dispatch AND the
    workflow-schedule claim in the same iteration, every 45 seconds, all day.
    This mirrors the pattern services/orion-vision-host/app/liveness.py already
    uses against this same endpoint.

    A journal is not worth failing a scheduler tick over: if notify is
    unreachable the daily entry is still written, just without this section.
    """
    def _blocking() -> list[dict[str, Any]]:
        import requests  # inside, so a missing dep degrades instead of aborting

        headers = {"X-Orion-Notify-Token": notify_api_token} if notify_api_token else {}
        url = f"{notify_url.strip().rstrip('/')}/attention"
        response = requests.get(
            url, params={"limit": limit}, headers=headers, timeout=timeout_sec
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            logger.warning(
                "capability_gap_attention_unexpected_payload type=%s", type(payload).__name__
            )
            return []
        rows = [i for i in payload if isinstance(i, dict)]
        # limit is orion-notify's hard ceiling (Query(le=200)); when the slice is
        # full the oldest row bounds how far back we can see, so say so rather
        # than silently reporting no gaps for a window we never covered.
        if len(rows) >= limit:
            logger.warning(
                "capability_gap_attention_slice_full limit=%d oldest=%s "
                "(older episodes may be invisible)",
                limit,
                rows[-1].get("created_at") if rows else None,
            )
        return rows

    try:
        import asyncio

        return await asyncio.to_thread(_blocking)
    except Exception as exc:
        logger.warning("capability_gap_attention_fetch_failed err=%s", exc)
        return []


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
