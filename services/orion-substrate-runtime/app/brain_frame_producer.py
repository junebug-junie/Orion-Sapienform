"""Pure assembly of SubstrateBrainFrameV1 from already-fetched substrate signals.

Deterministic and dependency-free (no DB, no bus): callers fetch the live graph,
lane health, latest self-state row, and latest attention broadcast, then hand
them here. Regions are computed from real activity and are the trackable spine;
node/edge samples are best-effort decoration with no continuity guarantee.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from orion.schemas.brain_frame import (
    BrainEdgeSampleV1,
    BrainNodeSampleV1,
    BrainRegionV1,
    BrainSpotlightV1,
    SubstrateBrainFrameV1,
)

_LANE_LABELS = {
    "biometrics": "Biometrics",
    "chat_grammar": "Chat grammar",
    "execution_trajectory": "Execution",
    "transport_bus": "Transport",
}


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _state_for(intensity: float, firing: float, starving: float) -> str:
    if intensity >= firing:
        return "firing"
    if intensity <= starving:
        return "starving"
    return "steady"


def _node_pressure(node: Any) -> float:
    md = getattr(node, "metadata", None) or {}
    val = md.get("dynamic_pressure")
    if val is None:
        val = md.get("prediction_error")
    return _clamp01(val or 0.0)


def _node_dormant(node: Any) -> bool:
    md = getattr(node, "metadata", None) or {}
    if md.get("dormant") is True:
        return True
    return _clamp01(getattr(node, "activation", 0.0)) <= 0.0


def _parse_dt(value: Any) -> datetime | None:
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _node_kind_regions(nodes, now, firing, starving) -> list[BrainRegionV1]:
    buckets: dict[str, list[float]] = {}
    for node in nodes:
        kind = str(getattr(node, "node_kind", "") or "unknown")
        buckets.setdefault(kind, []).append(_clamp01(getattr(node, "activation", 0.0)))
    regions: list[BrainRegionV1] = []
    for kind, activations in sorted(buckets.items()):
        intensity = max(activations) if activations else 0.0
        regions.append(
            BrainRegionV1(
                dimension="node_kind",
                region_id=f"node_kind:{kind}",
                label=kind.replace("_", " ").title(),
                intensity=intensity,
                state=_state_for(intensity, firing, starving),
                node_count=len(activations),
                as_of=now,
                stale=False,
                detail={"mean_activation": sum(activations) / len(activations) if activations else 0.0},
            )
        )
    return regions


def _lane_regions(lane_health: Mapping[str, Any], now, firing, starving) -> list[BrainRegionV1]:
    lag = dict(lane_health.get("cursor_lag_by_reducer") or {})
    backlog = dict(lane_health.get("pending_backlog_by_reducer") or {})
    quarantine = dict(lane_health.get("quarantine_by_reducer") or {})
    regions: list[BrainRegionV1] = []
    lane_keys = set(lag) | set(backlog) | set(_LANE_LABELS)
    for lane in sorted(lane_keys):
        lag_sec = float(lag.get(lane, 0.0) or 0.0)
        pending = float(backlog.get(lane, 0.0) or 0.0)
        # Fresh + moving lane = firing; stale/lagged lane = starving.
        freshness = 1.0 if lag_sec <= 60 else max(0.0, 1.0 - (lag_sec - 60) / 240.0)
        activity = min(1.0, pending / 20.0)
        intensity = _clamp01(0.6 * activity + 0.4 * freshness) if lag_sec <= 60 else _clamp01(freshness)
        if lag_sec > 300:
            intensity = min(intensity, starving)
        regions.append(
            BrainRegionV1(
                dimension="lane",
                region_id=f"lane:{lane}",
                label=_LANE_LABELS.get(lane, lane.replace("_", " ").title()),
                intensity=intensity,
                state=_state_for(intensity, firing, starving),
                node_count=int(pending),
                as_of=now,
                stale=False,
                detail={"lag_sec": lag_sec, "backlog": pending, "quarantine": float(quarantine.get(lane, 0) or 0)},
            )
        )
    return regions


def _self_state_regions(self_state, now, cadence_sec) -> list[BrainRegionV1]:
    if not isinstance(self_state, Mapping):
        return []
    as_of = _parse_dt(self_state.get("generated_at")) or now
    stale = (now - as_of).total_seconds() > cadence_sec
    dims = self_state.get("dimensions") or {}
    regions: list[BrainRegionV1] = []
    for dim_id, payload in sorted(dims.items()):
        if not isinstance(payload, Mapping):
            continue
        score = _clamp01(payload.get("score", 0.0))
        conf = _clamp01(payload.get("confidence", 0.0))
        regions.append(
            BrainRegionV1(
                dimension="self_state",
                region_id=f"self_state:{dim_id}",
                label=dim_id.replace("_", " ").title(),
                intensity=score,
                # self-state dims are always shown as steps, not fired/starved.
                state="steady",
                node_count=0,
                as_of=as_of,
                stale=stale,
                detail={"confidence": conf},
            )
        )
    return regions


def _spotlight(attention, now, cadence_sec) -> BrainSpotlightV1 | None:
    if attention is None:
        return None
    as_of = _parse_dt(getattr(attention, "generated_at", None)) or now
    return BrainSpotlightV1(
        attended_node_ids=[str(x) for x in getattr(attention, "attended_node_ids", []) or []],
        dwell_ticks=int(getattr(attention, "dwell_ticks", 0) or 0),
        coalition_stability=_clamp01(getattr(attention, "coalition_stability_score", 1.0)),
        description=getattr(attention, "selected_description", None),
        as_of=as_of,
        stale=(now - as_of).total_seconds() > cadence_sec,
    )


def _samples(nodes, edges, max_nodes, max_edges) -> tuple[list[BrainNodeSampleV1], list[BrainEdgeSampleV1]]:
    ranked = sorted(nodes, key=lambda n: _clamp01(getattr(n, "activation", 0.0)), reverse=True)
    node_samples = [
        BrainNodeSampleV1(
            node_id=str(getattr(n, "node_id", "") or ""),
            node_kind=str(getattr(n, "node_kind", "") or "unknown"),
            activation=_clamp01(getattr(n, "activation", 0.0)),
            pressure=_node_pressure(n),
            dormant=_node_dormant(n),
            label=str(getattr(n, "label", "") or "")[:120],
        )
        for n in ranked[: max(0, int(max_nodes))]
        if str(getattr(n, "node_id", "") or "")
    ]
    kept_ids = {s.node_id for s in node_samples}
    edge_samples: list[BrainEdgeSampleV1] = []
    max_e = max(0, int(max_edges))
    for e in edges or []:
        if len(edge_samples) >= max_e:
            break
        src = str(getattr(e, "src", None) or getattr(e, "source", "") or "")
        dst = str(getattr(e, "dst", None) or getattr(e, "target", "") or "")
        if not src or not dst or src not in kept_ids or dst not in kept_ids:
            continue
        edge_samples.append(
            BrainEdgeSampleV1(src=src, dst=dst, weight=_clamp01(getattr(e, "weight", 0.0) or 0.0))
        )
    return node_samples, edge_samples


def assemble_brain_frame(
    *,
    nodes: Iterable[Any],
    edges: Iterable[Any],
    lane_health: Mapping[str, Any],
    self_state: Mapping[str, Any] | None,
    attention: Any | None,
    settings: Any,
    now: datetime,
    tick_seq: int,
) -> SubstrateBrainFrameV1:
    nodes = list(nodes)
    firing = float(settings.brain_frame_firing_threshold)
    starving = float(settings.brain_frame_starving_threshold)

    regions = (
        _node_kind_regions(nodes, now, firing, starving)
        + _lane_regions(lane_health or {}, now, firing, starving)
        + _self_state_regions(self_state, now, float(settings.brain_frame_self_state_cadence_sec))
    )
    node_samples, edge_samples = _samples(
        nodes, list(edges), settings.brain_frame_sample_nodes, settings.brain_frame_sample_edges
    )

    max_activation = max((_clamp01(getattr(n, "activation", 0.0)) for n in nodes), default=0.0)
    phase = "live" if max_activation > 0.0 else "warming"

    frame_id = hashlib.sha256(f"{now.isoformat()}|{tick_seq}".encode("utf-8")).hexdigest()[:24]
    return SubstrateBrainFrameV1(
        frame_id=frame_id,
        generated_at=now,
        tick_seq=int(tick_seq),
        phase=phase,
        regions=regions,
        spotlight=_spotlight(attention, now, float(settings.brain_frame_spotlight_cadence_sec)),
        nodes=node_samples,
        edges=edge_samples,
    )
