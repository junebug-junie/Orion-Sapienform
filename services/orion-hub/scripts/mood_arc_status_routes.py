"""Read-only Hub API for the Mood Arc / Field Anomaly operator page.

Four views:

- GET /live: relays orion-field-digester's own `/health` status for its
  mood-arc encoder + live-enrichment coverage (app/anomaly_scorer.py
  ::status() in that service) -- this is genuinely that service's own
  runtime state, so it's fetched from the source rather than reconstructed
  from Postgres or the model directory's filesystem layout (see the
  `hub-mood-arc-status-ekg-traceability` plan's Approach section for why).
- GET /phi-v2-inventory: honest current-state snapshot of phi-v2's stubs --
  the two dead legacy `orion/inner_state_registry.py` entries plus whether
  the design doc's real (but unwired) successor pieces exist on disk. No
  progress bar, no fabricated completion percentage: phi-v2 itself is not
  implemented, and this says so plainly rather than implying otherwise.
- GET /inference-trace: the actual cockpit data -- recon_loss vs. the
  encoder's own live threshold over a real window, PLUS one real raw input
  channel's own trace over the same window, so the two can be plotted on
  one timeline and you can SEE the correlation, not just a stat tile.
  Reuses `substrate_brain_frame_log` (recon_loss/threshold/anomalous,
  already persisted by `_field_anomaly_regions()`) and
  `orion.field.pressure.collect_field_channel_pressures()` (the same merge
  every other field-pressure consumer reads) directly against
  `substrate_field_state` -- no new tables, no new schema.
- GET /downstream-triggers: real `telemetry_anomaly` metacog-trigger firings
  from `orion-equilibrium-service` (Postgres `metacog_trigger`, written by
  orion-sql-writer) in the same window -- the actual causal effect of this
  signal, not a claim about what it's "wired to". Confirmed live 2026-09-04:
  123 real fires in the preceding 24h.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from scripts.field_digester_client import FieldDigesterClientError, fetch_health
from scripts.service_logs import resolve_repo_root

router = APIRouter(prefix="/api/mood-arc-status", tags=["mood-arc-status"])

# The one raw input channel to overlay against recon_loss on the inference
# trace chart. Not configurable/generic (yet) -- deliberately picked from
# real data, not guessed: `top_channels` on the 3 most recent live
# telemetry_anomaly fires (2026-09-04) all named failure_pressure as the
# dominant driver (`docker exec ... metacog_trigger.upstream->>'top_channels'`).
# If that stops being true, this constant is the one place to change it, not
# a reason to build a full N-channel picker for a first cut.
_CORRELATED_CHANNEL = "failure_pressure"

_TRACE_ENGINE: Any = None
_TRACE_ENGINE_LOCK = threading.Lock()


def _trace_engine():
    """Own lazy-cached engine, same pattern (and same PR #2010
    connection-exhaustion rationale) as field_channel_glossary_routes.py/
    self_brain_routes.py -- not shared with either since this module has no
    existing engine to reuse and each of those two is already scoped to its
    own panel's polling cadence."""
    global _TRACE_ENGINE
    import os

    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return None
    with _TRACE_ENGINE_LOCK:
        if _TRACE_ENGINE is None:
            from sqlalchemy import create_engine

            _TRACE_ENGINE = create_engine(
                uri, pool_pre_ping=True, connect_args={"connect_timeout": 2}
            )
        return _TRACE_ENGINE

# `Path(__file__).resolve().parents[N]` breaks inside the Hub's own Docker
# image -- confirmed live (2026-09-04 docker up smoke test): the Dockerfile
# COPYs services/orion-hub flattened straight to /app (no `services/`,
# `docs/`, or repo-root level above it in the container), so a fixed
# `parents[3]` raised IndexError on every startup and crashed the whole
# service. `resolve_repo_root()` (scripts/service_logs.py) is the existing,
# already-used-by-two-other-routes mechanism for this exact problem: reads
# `ORION_REPO_ROOT` (the read-only `/repo` bind mount docker-compose.yml
# already sets up for grammar_atlas_routes.py/service_logs.py), falls back
# to walking up from this file, then cwd, then a bare `/repo` guess.
def _phi_v2_design_doc() -> Path:
    return resolve_repo_root() / "docs" / "superpowers" / "specs" / "2026-08-21-phi-v2-design.md"


def _phi_encoder_cli() -> Path:
    return resolve_repo_root() / "scripts" / "fit_phi_encoder.py"


# Both confirmed dead (2026-09-04 investigation): orion-spark-introspector,
# their shared producer, was deleted outright 2026-07-28. Hand-picked rather
# than a prefix scan over REGISTRY -- there are exactly two phi-tagged
# entries today and a silent new one showing up unlisted here is a real,
# worth-noticing gap, not something to paper over with a "phi" substring
# match that could also snag an unrelated future signal.
PHI_V2_LEGACY_SIGNAL_IDS: tuple[str, ...] = ("phi_heuristic.valence", "phi_intrinsic_reward.v1")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _first_sentences(text: str, count: int = 2) -> str:
    text = " ".join(text.split())  # collapse the registry's wrapped multi-line notes
    parts = _SENTENCE_SPLIT.split(text)
    return " ".join(parts[:count]).strip()


@router.get("/live")
async def live() -> dict[str, Any]:
    try:
        health = await fetch_health()
    except FieldDigesterClientError as exc:
        return {"reachable": False, "error": str(exc)}
    # `.get(..., default) or default`, not just `.get(..., default)`: review
    # finding (2026-09-04) -- the former only substitutes on a MISSING key,
    # so a hypothetical `"field_channel_anomaly": null` response would pass
    # None to `**`, raising TypeError instead of degrading gracefully.
    anomaly_block = health.get("field_channel_anomaly") or {"enabled": False}
    return {"reachable": True, **anomaly_block}


def _design_doc_status(doc_path: Path) -> str | None:
    if not doc_path.exists():
        return None
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Status:"):
            return line.removeprefix("Status:").strip()
    return None


@router.get("/phi-v2-inventory")
async def phi_v2_inventory() -> dict[str, Any]:
    from orion.inner_state_registry import REGISTRY

    repo_root = resolve_repo_root()
    signals = []
    by_id = {sig.signal_id: sig for sig in REGISTRY}
    for signal_id in PHI_V2_LEGACY_SIGNAL_IDS:
        sig = by_id.get(signal_id)
        if sig is None:
            signals.append({"signal_id": signal_id, "found_in_registry": False})
            continue
        # composition_status intentionally does NOT encode "retired" in this
        # registry's own convention (orion/inner_state_registry.md's
        # `field_attention_frame.v1` entry: RETIRED 2026-08-21 but stayed
        # COMPOSED) -- prose in `notes` carries that instead, and notes only
        # ever grow by dated append, so the most recent correction can land
        # well past what `last_note`'s first-2-sentences shows. A live,
        # code-verified check is the un-stale-able signal: does the claimed
        # producer_service still exist as a REAL, deployable service.
        #
        # NOT a bare `producer_dir.is_dir()` check -- confirmed live
        # (2026-09-04 docker smoke test) that `services/orion-spark-
        # introspector/` still physically exists on disk (app/, tests/,
        # train/, a gitignored .env) even though it was fully deleted from
        # git 2026-07-28 (commit 442e51ee2): `git rm` / the retirement PR
        # removed it from tracking but never `rm -rf`'d the directory
        # itself, so a bare-directory check reported "producer present" for
        # a service that is, by this repo's own convention, dead. Every
        # real service in this repo has its own docker-compose.yml (grepped
        # across services/*/); the leftover has none -- that's the reliable
        # signal, not mere directory presence.
        producer_dir = repo_root / "services" / sig.producer_service
        signals.append(
            {
                "signal_id": sig.signal_id,
                "found_in_registry": True,
                "producer_service": sig.producer_service,
                "producer_service_exists": (producer_dir / "docker-compose.yml").is_file(),
                "composition_status": sig.composition_status.value,
                "cognition_consumers": list(sig.cognition_consumers),
                "last_note": _first_sentences(sig.notes),
            }
        )

    design_doc = _phi_v2_design_doc()
    return {
        "legacy_signals": signals,
        "design_doc": {
            "path": str(design_doc.relative_to(repo_root)),
            "exists": design_doc.exists(),
            "status": _design_doc_status(design_doc),
        },
        "manual_cli_exists": _phi_encoder_cli().exists(),
    }


_MAX_TRACE_MINUTES = 180
_DEFAULT_TRACE_MINUTES = 30


@router.get("/inference-trace")
async def inference_trace(
    minutes: int = Query(default=_DEFAULT_TRACE_MINUTES, ge=1, le=_MAX_TRACE_MINUTES),
) -> dict[str, Any]:
    """Off the event loop: real synchronous SQLAlchemy + JSON decode over
    up to 180 minutes of ticks, same rationale as self_brain_routes.py's
    /frames/tail (measured 3s inline there before it was moved to a
    worker thread)."""
    return await asyncio.to_thread(_inference_trace_sync, minutes)


def _inference_trace_sync(minutes: int) -> dict[str, Any]:
    empty = {"minutes": minutes, "points": [], "channel": _CORRELATED_CHANNEL, "channel_points": []}
    engine = _trace_engine()
    if engine is None:
        return empty
    from sqlalchemy import text

    points: list[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT frame_json FROM substrate_brain_frame_log "
                    "WHERE generated_at >= NOW() - (:minutes * INTERVAL '1 minute') "
                    "ORDER BY generated_at ASC"
                ),
                {"minutes": minutes},
            )
            for (frame_json,) in rows:
                frame = frame_json if isinstance(frame_json, dict) else json.loads(frame_json)
                for region in frame.get("regions", []):
                    if region.get("region_id") != "field_anomaly:reconstruction":
                        continue
                    detail = region.get("detail") or {}
                    if "recon_loss" not in detail:
                        continue
                    points.append(
                        {
                            "t": region.get("as_of"),
                            "recon_loss": detail["recon_loss"],
                            "threshold": detail.get("threshold"),
                            "anomalous": bool(detail.get("anomalous")),
                        }
                    )
    except Exception:
        return empty

    channel_points = _correlated_channel_series_sync(engine, minutes)
    return {
        "minutes": minutes,
        "points": points,
        "channel": _CORRELATED_CHANNEL,
        "channel_points": channel_points,
    }


def _correlated_channel_series_sync(engine: Any, minutes: int) -> list[dict[str, Any]]:
    """One raw channel's own value over the window, pulled from
    substrate_field_state via the SAME merge function every other
    field-pressure consumer reads (orion.field.pressure
    ::collect_field_channel_pressures) -- not a re-derivation of channel
    polarity/merge logic, and not the full 38-channel glossary machinery
    field_channel_glossary_routes.py's build_channel_series() runs (that
    exists to classify EVERY known channel; this needs exactly one)."""
    from sqlalchemy import text

    from orion.field.pressure import collect_field_channel_pressures
    from orion.schemas.field_state import FieldStateV1

    out: list[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT field_json, generated_at FROM substrate_field_state "
                    "WHERE generated_at >= NOW() - (:minutes * INTERVAL '1 minute') "
                    "ORDER BY generated_at ASC"
                ),
                {"minutes": minutes},
            )
            for field_json, generated_at in rows:
                payload = field_json if isinstance(field_json, dict) else json.loads(field_json)
                try:
                    state = FieldStateV1.model_validate(payload)
                except Exception:  # noqa: BLE001 - skip unparsable historical rows
                    continue
                merged, _ = collect_field_channel_pressures(state)
                if _CORRELATED_CHANNEL not in merged:
                    continue
                out.append(
                    {
                        "t": generated_at.isoformat() if hasattr(generated_at, "isoformat") else generated_at,
                        "value": float(merged[_CORRELATED_CHANNEL]),
                    }
                )
    except Exception:
        return []
    return out


@router.get("/downstream-triggers")
async def downstream_triggers(
    minutes: int = Query(default=_DEFAULT_TRACE_MINUTES, ge=1, le=_MAX_TRACE_MINUTES),
) -> dict[str, Any]:
    return await asyncio.to_thread(_downstream_triggers_sync, minutes)


def _downstream_triggers_sync(minutes: int) -> dict[str, Any]:
    empty = {"minutes": minutes, "triggers": []}
    engine = _trace_engine()
    if engine is None:
        return empty
    from sqlalchemy import text

    triggers: list[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT timestamp, upstream FROM metacog_trigger "
                    "WHERE trigger_kind = 'telemetry_anomaly' "
                    "AND timestamp >= NOW() - (:minutes * INTERVAL '1 minute') "
                    "ORDER BY timestamp ASC"
                ),
                {"minutes": minutes},
            )
            for timestamp, upstream in rows:
                up = upstream if isinstance(upstream, dict) else (json.loads(upstream) if upstream else {})
                top_channels = up.get("top_channels") or []
                triggers.append(
                    {
                        "t": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
                        "recon_loss": up.get("recon_loss"),
                        "threshold": up.get("threshold"),
                        "deviation_direction": up.get("deviation_direction"),
                        # "channel=mse" strings as fit_encoder.top_channel_attribution
                        # produces them -- passed through as-is, not re-parsed.
                        "top_channel": top_channels[0] if top_channels else None,
                    }
                )
    except Exception:
        return empty
    return {"minutes": minutes, "triggers": triggers}
