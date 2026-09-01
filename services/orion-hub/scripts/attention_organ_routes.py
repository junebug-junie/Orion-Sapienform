"""Read-only Hub API for the Attention Organ operator tab.

Surfaces, in one place, the three live pieces of the precision-weighted
attention arc described in
``docs/superpowers/specs/2026-07-28-precision-weighted-attention-organ-and-
heartbeat-discrimination-design.md``:

1. **The organ** -- ``orion-heartbeat``'s N-trajectory tensor-network
   dissipation ensemble (``/h1`` + ``/health``): ensemble mean ratio, the
   per-trajectory ratios behind it, cross-trajectory spread, the verdict
   band edges it classifies with, intake/backpressure counters, and the real
   ``bus_synaptic``-driven reheat inputs the dissipation loop last used.
2. **The consumer** -- AST/HOT's ``AttentionSelfModelV1``, read from the
   durable ``substrate_attention_self_model`` table that
   ``orion-substrate-runtime``'s ``_attention_self_model_tick()`` writes,
   including the three ``heartbeat_*`` fields that carry (1) into it.
3. **The problem the arc is about** -- each Active-Inference domain's live
   ``prediction_error``, read straight off its ``node:substrate.<domain>``
   node in the substrate graph. This is the same value
   ``_brain_frame_prediction_error_by_domain()`` feeds the reducer, so the
   per-domain bars and the row's own ``prediction_error_confidence``
   reconcile exactly (verified live 2026-07-30: five domain values whose
   mean is exactly ``1 - prediction_error_confidence``).

Everything here is a read. No writes, no bus publishes, no new channel or
schema. Nothing this module returns is consumed by any cognition path -- it
exists so an operator can *see* whether the organ discriminates and whether
its signal is reaching AST/HOT, which is precisely what the design doc's
Acceptance Checks 1 and 3 ask for and which had no human-visible surface.

**Deliberately not computed here**: nothing is re-derived from a mirrored
copy of a constant that lives in another service. Ensemble verdict edges
(``low_ratio`` silence floor, ``high_ratio`` redundant conjunct,
``std_mixed`` / ``std_redundant_max``, ``bulk_low`` / ``bulk_redundant_min``)
and the reheat inputs (``raw_mean_abs_gap_zscore`` -> ``signal`` ->
``reheat_prob``) are read from ``orion-heartbeat``'s own ``/health`` payload,
which reports the values that service actually used. Mean-ratio color bands
are not the classifier — ``classify_ensemble_verdict()`` keys on std and
bulk after the silence floor. A number this tab shows is a number some real
producer actually produced.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

from orion.graph.falkor_client import RedisGraphQueryClient
from orion.substrate.attention_self_model import ACTIVE_INFERENCE_DOMAINS

from app.settings import settings

router = APIRouter(prefix="/api/attention-organ", tags=["attention-organ"])

# `node:substrate.<domain>` is the fixed identity
# `_write_prediction_error_node()` upserts per domain
# (services/orion-substrate-runtime/app/worker.py). Domains outside
# ACTIVE_INFERENCE_DOMAINS are still fetched and returned, flagged
# `active=False`, precisely so a retired instrument that is still ticking is
# visible rather than hidden -- CLAUDE.md's "a narrow, known-bad instrument
# merely excluded from one consumer is not retired, it is hiding".
# `codebase` added 2026-07-31 (docs/superpowers/specs/2026-07-30-codebase-
# mass-signal-design.md) once node:substrate.codebase started ticking live
# with real data (SUBSTRATE_WRITE_CODEBASE_PREDICTION_ERROR_NODE flipped on
# the same day) -- deliberately still excluded from ACTIVE_INFERENCE_DOMAINS
# below (orion/substrate/attention_self_model.py), same "sibling domain node,
# not yet folded into the confidence aggregate" status bus_synaptic held for
# a while before that was a separate, explicit decision.
#
# `transport` REMOVED 2026-07-31, when the domain was fully retired (producer,
# reducer map entry, function, and the live node). Leaving it listed puts this
# surface in one of two wrong states and there is no third: while the stale node
# still existed it rendered under the "still ticking, still readable elsewhere"
# heading -- asserting a retired domain is live -- and once the node is gone it
# renders a permanent red "no node" badge, which this module's own comment below
# calls "a different, louder failure", for a domain nobody should be alarmed
# about. A retired domain belongs off the list, not in a warning state forever.
#
# `harness_closure` deliberately STAYS: it is excluded from the aggregate but
# has a live producer, which is exactly what that heading is for.
KNOWN_PREDICTION_ERROR_DOMAINS: tuple[str, ...] = (
    "biometrics",
    "bus_synaptic",
    "chat",
    "codebase",
    "execution",
    "route",
    "harness_closure",
)

_NODE_ID_PREFIX = "node:substrate."

ALLOWED_HISTORY_MINUTES: frozenset[int] = frozenset({15, 60, 360, 1440})
DEFAULT_HISTORY_MINUTES: int = 60
HISTORY_ROW_CAP: int = 4000

_engine_instance: Any = None
_graph_client_instance: Any = None


def _engine():
    global _engine_instance
    if _engine_instance is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
        _engine_instance = create_engine(uri, pool_pre_ping=True)
    return _engine_instance


def _substrate_graph_client() -> RedisGraphQueryClient:
    """Module-level cached, matching _engine() above. RedisGraphQueryClient
    builds its own redis.Redis + ConnectionPool and exposes no close(), so a
    per-request client meant a fresh TCP connect/teardown on every poll --
    ~30/minute at the tab's fastest cadence, for a query returning 7 rows.
    Not a leak (refcounting drops the sockets), but pointless churn against a
    surface designed to be polled continuously.
    """
    global _graph_client_instance
    uri = (settings.FALKORDB_URI or "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="falkordb_uri_not_configured")
    if _graph_client_instance is None:
        graph_name = (
            getattr(settings, "FALKORDB_SUBSTRATE_GRAPH", "") or "orion_substrate"
        )
        _graph_client_instance = RedisGraphQueryClient(uri=uri, graph_name=graph_name)
    return _graph_client_instance


# --------------------------------------------------------------------------
# Pure helpers (no I/O) -- separated so the parsing/aggregation logic is
# testable against fixed rows, matching this repo's established route-module
# convention (see bus_synaptic_graph_routes.propagate_from_organ).
# --------------------------------------------------------------------------


def parse_predicted_shift_domain(predicted_shift: str | None) -> str | None:
    """Recover which domain won a tick's `predicted_shift`.

    `reduce_attention_self_model()` renders this field as a prose sentence
    (`"<domain> prediction-error rising (trend=...)"`) -- the winning domain
    is not persisted as its own column, so a rolling domain-dominance tally
    has to read it back off the front of that string. Kept narrow and
    anchored on the literal ` prediction-error ` marker the reducer emits
    rather than a loose split, so an unrelated future `predicted_shift`
    phrasing returns None (honestly "unknown") instead of silently
    contributing a garbage domain name to the tally.
    """
    if not predicted_shift or not isinstance(predicted_shift, str):
        return None
    marker = " prediction-error "
    idx = predicted_shift.find(marker)
    if idx <= 0:
        return None
    candidate = predicted_shift[:idx].strip()
    if not candidate or " " in candidate:
        return None
    return candidate


def summarize_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll a window of persisted AttentionSelfModelV1 payloads into the two
    distributions the design doc's acceptance checks are stated in terms of.

    - `verdict_counts`: how often each heartbeat verdict band fired. Acceptance
      Check 1 is literally "the verdict takes more than one value" -- this is
      that check, computed live rather than by eyeballing container logs.
    - `domain_counts`: how often each domain won `predicted_shift`. Acceptance
      Check 3 asks for this distribution to be materially less dominated by
      1-2 domains after any precision-weighting fix; showing it live is the
      before-picture that check compares against.

    `null_*` counters are kept explicitly rather than dropped so "this window
    had no heartbeat data" is visibly different from "the band never fired".
    """
    series: list[dict[str, Any]] = []
    verdict_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    null_verdict = 0
    null_domain = 0
    malformed = 0

    for row in rows:
        payload = row.get("self_model_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                malformed += 1
                continue
        if not isinstance(payload, dict):
            malformed += 1
            continue

        verdict = payload.get("heartbeat_verdict")
        if isinstance(verdict, str) and verdict:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        else:
            null_verdict += 1

        domain = parse_predicted_shift_domain(payload.get("predicted_shift"))
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        else:
            null_domain += 1

        generated_at = row.get("generated_at")
        series.append(
            {
                "generated_at": (
                    generated_at.isoformat()
                    if isinstance(generated_at, datetime)
                    else str(generated_at)
                ),
                "heartbeat_mean_ratio": payload.get("heartbeat_mean_ratio"),
                "heartbeat_std_ratio": payload.get("heartbeat_std_ratio"),
                "heartbeat_bulk_penetration_depth": payload.get(
                    "heartbeat_bulk_penetration_depth"
                ),
                "heartbeat_verdict": verdict if isinstance(verdict, str) else None,
                "prediction_error_confidence": payload.get("prediction_error_confidence"),
                "predicted_shift_domain": domain,
            }
        )

    return {
        "sample_count": len(series),
        "series": series,
        "verdict_counts": verdict_counts,
        "verdict_distinct_values": len(verdict_counts),
        "null_verdict_count": null_verdict,
        "domain_counts": domain_counts,
        "null_domain_count": null_domain,
        # Same argument as the null_* counters above, applied to rows that
        # could not be read at all: a window where every row failed to parse
        # must not look identical to a window with no activity.
        "malformed_row_count": malformed,
    }


def build_domain_rows(
    graph_rows: list[dict[str, Any]], *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Shape raw `node:substrate.<domain>` graph rows into per-domain entries.

    `active` marks membership in `ACTIVE_INFERENCE_DOMAINS` -- the set
    `_unconditional_prediction_error_confidence()` actually averages over.
    Inactive domains are still returned (see KNOWN_PREDICTION_ERROR_DOMAINS'
    comment) so an excluded-but-still-ticking instrument stays visible.

    `at_ceiling`/`at_floor` are flagged rather than silently rendered as
    ordinary values: CLAUDE.md's metric quality gate step 4 treats both a
    pinned 1.0 and a decayed-to-0.0 reading as suspect-until-checked, and
    both failure modes have real incident history on these exact nodes
    (bus_synaptic frozen at 1.0, PR #1449; route decayed toward 0.0 by a
    generic staleness loop, 2026-07-26).
    """
    now = now or datetime.now(timezone.utc)
    by_domain: dict[str, dict[str, Any]] = {}

    for row in graph_rows:
        node_id = str(row.get("node_id") or "")
        if not node_id.startswith(_NODE_ID_PREFIX):
            continue
        domain = node_id[len(_NODE_ID_PREFIX) :]
        raw_value = row.get("prediction_error")
        try:
            value = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            value = None

        observed_at = row.get("observed_at")
        observed_age_sec: float | None = None
        if isinstance(observed_at, str) and observed_at:
            try:
                parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                observed_age_sec = round((now - parsed).total_seconds(), 1)
            except ValueError:
                observed_age_sec = None

        try:
            activation = float(row["activation"]) if row.get("activation") is not None else None
        except (TypeError, ValueError):
            activation = None

        by_domain[domain] = {
            "domain": domain,
            "node_id": node_id,
            "prediction_error": value,
            "active": domain in ACTIVE_INFERENCE_DOMAINS,
            "activation": activation,
            "observed_at": observed_at if isinstance(observed_at, str) else None,
            "observed_age_sec": observed_age_sec,
            "at_ceiling": value is not None and value >= 1.0,
            "at_floor": value is not None and value <= 0.0,
            "present": True,
        }

    out: list[dict[str, Any]] = []
    for domain in KNOWN_PREDICTION_ERROR_DOMAINS:
        if domain in by_domain:
            out.append(by_domain.pop(domain))
        else:
            # A known domain with no node at all is a different, louder
            # failure than a node reading 0.0 -- never collapse them.
            out.append(
                {
                    "domain": domain,
                    "node_id": f"{_NODE_ID_PREFIX}{domain}",
                    "prediction_error": None,
                    "active": domain in ACTIVE_INFERENCE_DOMAINS,
                    "activation": None,
                    "observed_at": None,
                    "observed_age_sec": None,
                    "at_ceiling": False,
                    "at_floor": False,
                    "present": False,
                }
            )
    # Any domain node that exists but isn't in the known list (a new producer
    # landed and this module wasn't updated) is appended rather than dropped.
    out.extend(by_domain.values())
    return out


# How much the 5-domain mean can move if exactly ONE active domain traverses
# its full [0, 1] range between the row's write and this module's graph read:
# 1.0 / len(ACTIVE_INFERENCE_DOMAINS). Confirmed live 2026-07-30 that this is
# not hypothetical -- `execution` and `bus_synaptic` both swing between ~0 and
# a pinned 1.0 across consecutive ~30s ticks, so a delta of ~0.2 on a row a few
# seconds old is ordinary sampling skew, not evidence of anything.
_SINGLE_DOMAIN_SWING = 1.0 / max(1, len(ACTIVE_INFERENCE_DOMAINS))

# Past this row age, the two readings are simply too far apart in time for the
# comparison to carry information either way, and asserting a verdict would be
# manufacturing one.
_RECONCILE_MAX_ROW_AGE_SEC = 60.0


def reconcile_confidence(
    domain_rows: list[dict[str, Any]],
    prediction_error_confidence: float | None,
    *,
    row_age_sec: float | None = None,
) -> dict[str, Any]:
    """Check the persisted aggregate against the live per-domain values.

    `_unconditional_prediction_error_confidence()` is
    `1 - mean(prediction_error over ACTIVE_INFERENCE_DOMAINS)`. Recomputing
    that from the domain nodes and diffing it against what the row actually
    stored is a live provenance check: a gap too large to be explained by the
    domains simply having moved means the two are no longer reading the same
    thing (an instrument is being clobbered, or the reducer's domain set has
    drifted from this module's).

    **The verdict is deliberately not a fixed-tolerance boolean.** The first
    version of this used `abs(delta) <= 0.05`, and live data immediately
    showed why that is a bad instrument: the row is written on
    substrate-runtime's ~30s tick while the graph read happens now, and two of
    the five domains genuinely swing across their whole range between ticks --
    so a flat tolerance reports "divergent" purely as a function of when the
    operator happened to poll. A check that fires on its own sampling schedule
    is noise, and shipping it would be the exact failure mode this tab exists
    to expose. The verdict is therefore stated against a model of how far the
    aggregate could legitimately have moved, and withheld entirely once the
    row is too old to compare against at all.
    """
    values = [
        r["prediction_error"]
        for r in domain_rows
        if r.get("active") and r.get("prediction_error") is not None
    ]
    if not values or prediction_error_confidence is None:
        return {
            "recomputed": None,
            "persisted": prediction_error_confidence,
            "delta": None,
            "domains_used": len(values),
            "row_age_sec": row_age_sec,
            "verdict": "unknown",
            "basis": "no active-domain values or no persisted aggregate to compare",
        }

    recomputed = 1.0 - (sum(values) / len(values))
    delta = recomputed - float(prediction_error_confidence)
    out = {
        "recomputed": round(recomputed, 4),
        "persisted": float(prediction_error_confidence),
        "delta": round(delta, 4),
        "domains_used": len(values),
        "row_age_sec": row_age_sec,
    }

    # Compared at the persisted value's OWN precision, not raw float equality:
    # the reducer rounds prediction_error_confidence to 4dp before storing it,
    # so an exact float match is unreachable by construction and "exact" would
    # be dead code.
    if round(delta, 4) == 0.0:
        out["verdict"] = "exact"
        out["basis"] = "row was written from the same domain values read here"
    elif row_age_sec is not None and row_age_sec > _RECONCILE_MAX_ROW_AGE_SEC:
        out["verdict"] = "unknown"
        out["basis"] = (
            f"row is {row_age_sec:.0f}s old -- too far from this graph read for the "
            "comparison to mean anything either way"
        )
    elif abs(delta) <= _SINGLE_DOMAIN_SWING:
        out["verdict"] = "within_drift"
        out["basis"] = (
            f"delta is within {_SINGLE_DOMAIN_SWING:.2f}, the most one of "
            f"{len(ACTIVE_INFERENCE_DOMAINS)} domains moving across its full range "
            "can shift the mean between the row's write and this read"
        )
    else:
        out["verdict"] = "divergent"
        out["basis"] = (
            f"delta exceeds {_SINGLE_DOMAIN_SWING:.2f} -- more than one domain would "
            "have had to move fully, which points at a source mismatch rather than timing"
        )
    return out


def normalize_history_minutes(minutes: int | None) -> int:
    if minutes in ALLOWED_HISTORY_MINUTES:
        return int(minutes)
    return DEFAULT_HISTORY_MINUTES


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def _heartbeat_get(path: str) -> dict[str, Any]:
    """One heartbeat debug-endpoint read. Never raises -- an unreachable
    heartbeat must render as an honest "organ offline" panel, not a 500 that
    also blanks the AST/HOT and per-domain panels next to it, which are
    served by entirely different backends and are still perfectly valid.
    """
    base = (settings.HUB_HEARTBEAT_BASE_URL or "").strip().rstrip("/")
    if not base:
        return {"ok": False, "error": "heartbeat_base_url_not_configured"}
    try:
        resp = requests.get(
            f"{base}{path}",
            timeout=float(settings.HUB_ATTENTION_ORGAN_TIMEOUT_SEC),
        )
        resp.raise_for_status()
        payload = resp.json()
        return {"ok": True, "payload": payload}
    except Exception as exc:  # noqa: BLE001 - reported to the operator, not raised
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _load_latest_self_model() -> tuple[dict[str, Any] | None, datetime | None]:
    with _engine().connect() as conn:
        row = (
            conn.execute(
                text(
                    """
                    SELECT generated_at, self_model_json
                    FROM substrate_attention_self_model
                    ORDER BY generated_at DESC
                    LIMIT 1
                    """
                )
            )
            .mappings()
            .first()
        )
    if not row:
        return None, None
    payload = row["self_model_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return (payload if isinstance(payload, dict) else None), row["generated_at"]


def _load_domain_rows() -> list[dict[str, Any]]:
    client = _substrate_graph_client()
    # Labelled (:SubstrateNode) rather than a bare MATCH (n): there is no
    # index on node_id (checked live -- CALL db.indexes() is empty), and this
    # is the substrate CONCEPT graph, designed to accumulate, scanned on every
    # poll. Free today at 10 nodes; not free later. STARTS WITH is kept over
    # an `IN $ids` form on purpose so a domain node this module does not yet
    # know about is still discovered (see build_domain_rows' tail).
    return client.graph_query(
        """
        MATCH (n:SubstrateNode)
        WHERE n.node_id STARTS WITH $prefix
        RETURN n.node_id AS node_id,
               n.prediction_error AS prediction_error,
               n.observed_at AS observed_at,
               n.activation AS activation
        """,
        {"prefix": _NODE_ID_PREFIX},
    )


@router.get("/snapshot")
def snapshot() -> dict[str, Any]:
    """One poll = one request. Each of the three sources degrades
    independently: any one being down leaves its own block flagged and the
    others intact.

    Deliberately `def`, not `async def`: this handler does blocking I/O (two
    `requests.get` calls, a Postgres read, a FalkorDB read) and the tab
    auto-polls it as fast as every 2s. Declared async, that would block Hub's
    single event loop -- chat, websockets, every other route -- for the full
    duration of every poll. FastAPI runs a sync handler in its threadpool
    instead. Some older read-only routes here are `async def` with the same
    blocking bodies; those are user-initiated one-shots, which is why the
    pattern survived. It is not safe at this cadence.
    """
    now = datetime.now(timezone.utc)

    h1 = _heartbeat_get("/h1")
    health = _heartbeat_get("/health")

    self_model: dict[str, Any] | None = None
    self_model_generated_at: datetime | None = None
    self_model_error: str | None = None
    try:
        self_model, self_model_generated_at = _load_latest_self_model()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        self_model_error = f"{type(exc).__name__}: {exc}"

    domain_rows: list[dict[str, Any]] = []
    domains_error: str | None = None
    try:
        domain_rows = build_domain_rows(_load_domain_rows(), now=now)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        domains_error = f"{type(exc).__name__}: {exc}"

    live_h1 = (h1.get("payload") or {}).get("h1") if h1.get("ok") else None

    # Is the heartbeat -> AST/HOT wire actually live right now? The row's own
    # heartbeat_basis string names the tick_count it was built from, so a
    # frozen wire shows up as a growing gap between that and the organ's
    # current tick_count -- a real staleness check, not just "the field is
    # non-null".
    link: dict[str, Any] = {
        "self_model_has_heartbeat": bool(self_model and self_model.get("heartbeat_verdict")),
        "self_model_mean_ratio": (self_model or {}).get("heartbeat_mean_ratio"),
        "live_mean_ratio": (live_h1 or {}).get("mean_ratio"),
        "self_model_basis": (self_model or {}).get("heartbeat_basis") or "",
        "live_tick_count": (live_h1 or {}).get("tick_count"),
    }
    if self_model_generated_at is not None:
        link["self_model_age_sec"] = round(
            (now - self_model_generated_at).total_seconds(), 1
        )
    else:
        link["self_model_age_sec"] = None

    return {
        "generated_at": now.isoformat(),
        "heartbeat": {
            # Both endpoints reachable AND an actual H1 reading present.
            # `h1.get("ok")` is only transport-level: /h1 answers 200 with
            # {"ok": false, "reason": "no_h1_computed_yet"} before the first
            # ensemble tick, which is precisely a degraded state the status
            # line must report, not a healthy one.
            "ok": bool(h1.get("ok") and health.get("ok") and live_h1 is not None),
            "h1": live_h1,
            "h1_error": None if h1.get("ok") else h1.get("error"),
            "h1_reason": (h1.get("payload") or {}).get("reason") if h1.get("ok") else None,
            "health": health.get("payload") if health.get("ok") else None,
            "health_error": None if health.get("ok") else health.get("error"),
        },
        "self_model": {
            "ok": self_model is not None,
            "error": self_model_error,
            "generated_at": (
                self_model_generated_at.isoformat() if self_model_generated_at else None
            ),
            "model": self_model,
        },
        "domains": {
            "ok": domains_error is None,
            "error": domains_error,
            "active_domains": sorted(ACTIVE_INFERENCE_DOMAINS),
            "rows": domain_rows,
            "reconciliation": reconcile_confidence(
                domain_rows,
                (self_model or {}).get("prediction_error_confidence"),
                row_age_sec=link["self_model_age_sec"],
            ),
        },
        "link": link,
    }


@router.get("/history")
def history(minutes: int = Query(DEFAULT_HISTORY_MINUTES)) -> dict[str, Any]:
    """Sync for the same reason as snapshot() -- blocking SQLAlchemy read."""
    window_minutes = normalize_history_minutes(minutes)
    with _engine().connect() as conn:
        # DESC + reverse (not ASC + LIMIT) so a window wide enough to hit the
        # cap keeps the NEWEST rows, not the oldest -- same reasoning as
        # field_channel_glossary_routes.health().
        result = (
            conn.execute(
                text(
                    """
                    SELECT generated_at, self_model_json
                    FROM substrate_attention_self_model
                    WHERE generated_at >= NOW() - (:minutes * INTERVAL '1 minute')
                    ORDER BY generated_at DESC
                    LIMIT :row_cap
                    """
                ),
                {"minutes": window_minutes, "row_cap": HISTORY_ROW_CAP},
            )
            .mappings()
            .all()
        )
    rows = [dict(r) for r in reversed(result)]
    summary = summarize_history(rows)
    return {
        "window_minutes": window_minutes,
        "row_cap": HISTORY_ROW_CAP,
        "truncated": len(rows) >= HISTORY_ROW_CAP,
        **summary,
    }
