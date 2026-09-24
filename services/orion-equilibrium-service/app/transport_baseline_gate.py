"""Transport baseline gate: per-hop EWMA baselines -> transport metacog triggers.

Service-side glue around the pure reducer ``orion/metacog/transport_baseline.py``
(spec ``docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-
baseline-design.md``, A1-A4). This module owns:

- building the reducer config from settings,
- (de)serializing reducer state for Redis,
- turning reducer events into ``MetacogTriggerV1`` per the transport OUTPUT
  CONTRACT the metacog row mapper builds against,
- the structured log lines the log-only phase is judged on.

Two flags, deliberately separate:
- ``EQUILIBRIUM_TRANSPORT_BASELINE_ENABLE`` (default true): fold every snapshot,
  persist state, log per-key z / ratio / calls / open conditions. Publishes
  nothing.
- ``EQUILIBRIUM_TRANSPORT_BASELINE_EMIT`` (default false): additionally publish
  the episode triggers. While emitting, the legacy rpc_health timeout branch is
  not called (the timeout / zero_success episodes replace it).

Keys matching ``EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS`` (default
``log_orion_metacognition``) are baselined and logged but never produce a
trigger -- metacog's own draft call cannot start another metacog draft.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, deque
from dataclasses import asdict
from typing import Any, Iterable

from orion.metacog.transport_baseline import (
    FoldResult,
    TransportBaselineConfig,
    TransportConditionEvent,
    fold_snapshot,
    load_state,
    new_state,
    state_to_dict,
)
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.equilibrium.transport_baseline_gate")

EVIDENCE_SOURCE = "transport_baseline"


def gate_from_settings(settings: Any) -> "TransportBaselineGate":
    return TransportBaselineGate(
        config_from_settings(settings),
        settings.transport_exclude_labels(),
        max_triggers_per_hour=int(settings.transport_baseline_max_triggers_per_hour),
    )


def config_from_settings(settings: Any) -> TransportBaselineConfig:
    return TransportBaselineConfig(
        min_calls=int(settings.transport_baseline_min_calls),
        n_warm=int(settings.transport_baseline_n_warm),
        spike_z=float(settings.transport_baseline_spike_z),
        min_excess_ms=float(settings.transport_baseline_min_excess_ms),
        saturation_ratio=float(settings.transport_baseline_saturation_ratio),
        regime_after_s=float(settings.transport_baseline_regime_after_sec),
    )


def _fmt(v: float | None, spec: str) -> str:
    return "na" if v is None else format(v, spec)


def build_transport_baseline_trigger(
    event: TransportConditionEvent,
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
) -> MetacogTriggerV1 | None:
    """Map one reducer event to a trigger. Excluded keys never map."""
    if event.excluded:
        return None
    if event.condition in ("spike", "saturation", "regime_shift"):
        detail = f"z={_fmt(event.z, '.2f')}:ratio={_fmt(event.saturation_ratio, '.2f')}"
    else:
        detail = f"timeouts={event.timeout_count}"
    reason = (
        f"transport:{event.condition}:{event.phase}:{event.service}:{event.key}:{detail}"
    )[:500]
    return MetacogTriggerV1(
        trigger_kind="transport",
        reason=reason,
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        signal_refs=[event.service, event.key],
        upstream={
            "evidence_source": EVIDENCE_SOURCE,
            "condition": event.condition,
            "phase": event.phase,
            "service": event.service,
            "instance": event.instance,
            "key": event.key,
            "z": event.z,
            "saturation_ratio": event.saturation_ratio,
            "baseline_ms": event.baseline_ms,
            "floor_ms": event.floor_ms,
            "window_mean_ms": event.window_mean_ms,
            "calls_per_min": event.calls_per_min,
            "calls_per_min_usual": event.calls_per_min_usual,
            "duration_s": event.duration_s,
            "peak_ms": event.peak_ms,
            "timeout_count": event.timeout_count,
        },
    )


class TransportBaselineGate:
    def __init__(
        self,
        config: TransportBaselineConfig,
        exclude_labels: Iterable[str],
        *,
        max_triggers_per_hour: int = 30,
    ) -> None:
        self.config = config
        self.exclude_labels = tuple(exclude_labels)
        self.state = new_state(config)
        self.max_triggers_per_hour = int(max_triggers_per_hour)
        self._published_ts: deque[float] = deque()
        self._skip_counts: Counter[str] = Counter()

    def admit(self, now_ts: float) -> bool:
        """Global publish budget for baseline triggers. They bypass the 30 s
        transport lane (episodes are self-limiting), so a mesh-wide outage
        across many hops must still not become a burst of metacog drafts.
        Over budget -> the caller logs and drops the trigger."""
        while self._published_ts and now_ts - self._published_ts[0] >= 3600.0:
            self._published_ts.popleft()
        if self.max_triggers_per_hour > 0 and len(self._published_ts) >= self.max_triggers_per_hour:
            return False
        self._published_ts.append(now_ts)
        return True

    def reset(self, reason: str) -> None:
        logger.error("transport_baseline cold_start reason=%s", reason)
        self.state = new_state(self.config)

    def load(self, raw: str | bytes | None) -> str | None:
        """Restore from the persisted JSON string. Returns the cold-start reason
        (None on a clean resume) and logs it -- a config change is refused, not
        silently merged."""
        data: Any = None
        if raw is not None:
            try:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                data = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                data = "malformed"
        self.state, reason = load_state(data, self.config)
        if reason is not None:
            logger.warning(
                "transport_baseline cold_start reason=%s fingerprint=%s", reason, self.config.fingerprint()
            )
        else:
            logger.info(
                "transport_baseline resumed keys=%d fingerprint=%s", len(self.state.keys), self.state.fingerprint
            )
        return reason

    def dump(self) -> str:
        return json.dumps(state_to_dict(self.state))

    def process(
        self,
        payload: dict[str, Any],
        *,
        zen_state: str,
        pressure: float,
        recall_enabled: bool,
    ) -> tuple[FoldResult, list[MetacogTriggerV1]]:
        result = fold_snapshot(
            self.state, payload, config=self.config, exclude_labels=self.exclude_labels
        )
        if result.skipped_reason is not None:
            # "No data" must stay distinguishable from "calm" in the log-only
            # week: count skips and log the first and every 100th per reason.
            self._skip_counts[result.skipped_reason] += 1
            n = self._skip_counts[result.skipped_reason]
            if n == 1 or n % 100 == 0:
                logger.info(
                    "transport_baseline_skip reason=%s count=%d service=%s",
                    result.skipped_reason,
                    n,
                    payload.get("service") if isinstance(payload, dict) else None,
                )
        triggers: list[MetacogTriggerV1] = []
        for ev in result.events:
            trig = build_transport_baseline_trigger(
                ev, zen_state=zen_state, pressure=pressure, recall_enabled=recall_enabled
            )
            if trig is not None:
                triggers.append(trig)
        return result, triggers


def log_fold_result(result: FoldResult, *, emit: bool) -> None:
    """One structured line per key with traffic, evaluation, or an open episode,
    plus one per condition event. These lines are what acceptance check 1 (the
    log-only week) is measured from."""
    for ob in result.observations:
        if not (ob.evaluated or ob.success_count or ob.timeout_count or ob.open_conditions):
            continue
        logger.info("transport_baseline_obs %s", json.dumps(asdict(ob), sort_keys=True))
    for ev in result.events:
        logger.info(
            "transport_baseline_event emit=%s %s",
            bool(emit and not ev.excluded),
            json.dumps(asdict(ev), sort_keys=True),
        )
