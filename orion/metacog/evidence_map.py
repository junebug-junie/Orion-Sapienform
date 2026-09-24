"""orion/metacog/evidence_map.py

Deterministic mapping from a metacog trigger's own upstream evidence into the
fields of its ``orion_metacog`` row (spec section B,
docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md).

Before this module, ``severity`` came from how sure the *writing model* felt
about its own paraphrase (a logprob probe) plus the metacog pipeline's own
failed-step count, and ``causal_density`` was a blend that only ever read 0 or
0.25. Neither looked at the event. Live 2026-09-24: transport rows marked
"nominal" had a *higher* median p95 (15.8 s) than rows marked "critical"
(10.2 s).

Here every field is a pure function of ``(trigger_kind, reason, upstream)``:

- ``severity``: from the event's own magnitude, per kind.
- ``magnitude``: 0..1, banded so severity and magnitude can never disagree:
  nominal -> [0, 0.3), degraded -> [0.3, 0.6), critical -> [0.6, 1.0].
  ``causal_density.score`` is this value, so ``is_causally_dense`` (score >=
  0.6) is exactly "critical".
- ``evidence``: short strings built from real upstream fields.
- ``touches``: the services / channels / artifacts named in upstream.
- ``summary_fallback``: a deterministic one-liner used when the LLM draft fails.

All thresholds below are PROVISIONAL starting values (spec section B table),
chosen from the live 7-day distributions noted beside each; the replay script
``scripts/analysis/replay_metacog_capture.py`` re-checks them against stored
triggers. Malformed or unknown upstream never raises: it maps to nominal,
magnitude 0, rationale ``no_evidence``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

Severity = Literal["nominal", "degraded", "critical"]

# Magnitude bands. Kept as named constants so the publish path, tests and the
# replay report all agree on what a band edge means.
DEGRADED_FLOOR = 0.3
CRITICAL_FLOOR = 0.6

_EVIDENCE_MAX_CHARS = 160
_MAX_EVIDENCE_ITEMS = 8


@dataclass(frozen=True)
class EvidenceMapping:
    severity: Severity
    magnitude: float
    density_label: str
    density_rationale: str
    evidence: list[str] = field(default_factory=list)
    touches: list[str] = field(default_factory=list)
    summary_fallback: str = ""

    @property
    def causal_density(self) -> dict[str, Any]:
        return {
            "label": self.density_label,
            "score": self.magnitude,
            "rationale": self.density_rationale,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "magnitude": self.magnitude,
            "causal_density": self.causal_density,
            "evidence": list(self.evidence),
            "touches": list(self.touches),
            "summary_fallback": self.summary_fallback,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Finite float or None. bool is rejected (True is not 1 ms of latency)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        try:
            f = float(value)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _int(value: Any) -> int | None:
    f = _num(value)
    return int(f) if f is not None else None


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def banded(x: float, degraded_at: float, critical_at: float, saturate_at: float) -> tuple[Severity, float]:
    """Map a raw, monotone "how bad" quantity onto (severity, magnitude).

    x <= 0              -> nominal, 0
    0 < x < degraded_at -> nominal,  magnitude in (0, 0.3)
    < critical_at       -> degraded, magnitude in [0.3, 0.6)
    >= critical_at      -> critical, magnitude in [0.6, 1.0], saturating at saturate_at

    One function decides both, so severity is monotone in magnitude by
    construction (spec acceptance check 4).
    """
    if not (0.0 < degraded_at < critical_at < saturate_at):
        raise ValueError("banded() needs 0 < degraded_at < critical_at < saturate_at")
    if x <= 0.0:
        return "nominal", 0.0
    if x < degraded_at:
        return "nominal", round(DEGRADED_FLOOR * x / degraded_at, 4)
    if x < critical_at:
        frac = (x - degraded_at) / (critical_at - degraded_at)
        return "degraded", round(DEGRADED_FLOOR + (CRITICAL_FLOOR - DEGRADED_FLOOR) * frac, 4)
    frac = _clip01((x - critical_at) / (saturate_at - critical_at))
    return "critical", round(CRITICAL_FLOOR + (1.0 - CRITICAL_FLOOR) * frac, 4)


_SEVERITY_RANK = {"nominal": 0, "degraded": 1, "critical": 2}


def severity_rank(severity: str) -> int:
    return _SEVERITY_RANK.get(severity, 0)


def _worst(*pairs: tuple[Severity, float]) -> tuple[Severity, float]:
    best: tuple[Severity, float] = ("nominal", 0.0)
    for sev, mag in pairs:
        if (severity_rank(sev), mag) > (severity_rank(best[0]), best[1]):
            best = (sev, mag)
    return best


def _cap(sev: Severity, mag: float, ceiling: Severity) -> tuple[Severity, float]:
    """Cap severity at `ceiling`, clamping magnitude into that band's top."""
    if severity_rank(sev) <= severity_rank(ceiling):
        return sev, mag
    top = DEGRADED_FLOOR if ceiling == "nominal" else CRITICAL_FLOOR
    return ceiling, round(top - 0.0001, 4)


def _nominal_band(x01: float) -> float:
    """Scale a 0..1 quantity into the nominal band [0, 0.3) for kinds that
    are informational, not trouble (flow, insight, episode close)."""
    return round(DEGRADED_FLOOR * 0.999 * _clip01(x01), 4)


def density_label(score: float) -> str:
    """Same cut points as orion/metacog/service.py::_label_for_score."""
    if score >= 0.85:
        return "critical"
    if score >= CRITICAL_FLOOR:
        return "dense"
    if score >= 0.25:
        return "salient"
    return "ambient"


def _clean(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        if len(s) > _EVIDENCE_MAX_CHARS:
            s = s[: _EVIDENCE_MAX_CHARS - 3] + "..."
        if s not in out:
            out.append(s)
    return out[:_MAX_EVIDENCE_ITEMS]


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        s = str(item or "").strip()
        if s and s not in out:
            out.append(s)
    return out


def _secs(ms: float | None) -> str:
    if ms is None:
        return "?"
    return f"{ms / 1000.0:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def _build(
    *,
    kind: str,
    severity: Severity,
    magnitude: float,
    basis: str,
    evidence: list[str],
    touches: list[str],
    headline: str,
) -> EvidenceMapping:
    magnitude = round(_clip01(magnitude), 4)
    return EvidenceMapping(
        severity=severity,
        magnitude=magnitude,
        density_label=density_label(magnitude),
        density_rationale=f"event_magnitude[{kind}]: {basis}",
        evidence=_clean(evidence),
        touches=_dedupe(touches),
        summary_fallback=f"{kind} ({severity}): {headline}"[:300],
    )


def _no_evidence(kind: str, reason: str, why: str = "no_evidence") -> EvidenceMapping:
    reason_s = str(reason or "").strip()
    return EvidenceMapping(
        severity="nominal",
        magnitude=0.0,
        density_label="ambient",
        density_rationale=f"no_evidence[{kind}]: {why}",
        evidence=_clean([f"reason={reason_s}"] if reason_s else []),
        touches=[],
        summary_fallback=f"{kind or 'unknown'} (nominal): no mappable evidence ({why})."[:300],
    )


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

# rpc_health_snapshot (today's pooled per-service window, 30 s).
# Timeouts lead (spec A3.4): 1 timeout -> degraded, >= 2 -> critical.
TRANSPORT_TIMEOUT_DEGRADED = 1.0
TRANSPORT_TIMEOUT_CRITICAL = 2.0
TRANSPORT_TIMEOUT_SATURATE = 6.0
# Latency: p95 / the gate's own latency_p95_threshold_ms (5 s today). The
# gate fires at 1x; live 7d median of fired rows is ~2-3x and is dominated by
# metacog's own 10-20 s background LLM calls (spec: "the self-loop"), so a
# latency-only row stays nominal until 3x and only reaches critical at 6x.
TRANSPORT_LATENCY_RATIO_DEGRADED = 3.0
TRANSPORT_LATENCY_RATIO_CRITICAL = 6.0
TRANSPORT_LATENCY_RATIO_SATURATE = 12.0
# Spec A3.2: a latency reading from fewer calls than this is "the slowest
# call", not a percentile -- it may say degraded, never critical.
TRANSPORT_MIN_CALLS_FOR_CRITICAL_LATENCY = 5

# bus_synaptic prediction error / its gate threshold (0.15). Live 7d: median
# 1.5x, max 4.9x.
BUS_SYNAPTIC_RATIO_DEGRADED = 2.0
BUS_SYNAPTIC_RATIO_CRITICAL = 3.0
BUS_SYNAPTIC_RATIO_SATURATE = 5.0

# transport_baseline (the new EWMA gate, spec A1-A4).
TRANSPORT_Z_DEGRADED = 3.0
TRANSPORT_Z_CRITICAL = 5.0
TRANSPORT_Z_SATURATE = 10.0
TRANSPORT_SATURATION_DEGRADED = 2.0
TRANSPORT_SATURATION_CRITICAL = 3.0
TRANSPORT_SATURATION_SATURATE = 6.0

_RPC_TIMEOUT_ELAPSED = re.compile(r"after\s+([0-9.]+)s")


def _timeout_band(count: int) -> tuple[Severity, float]:
    return banded(
        float(max(count, 0)),
        TRANSPORT_TIMEOUT_DEGRADED,
        TRANSPORT_TIMEOUT_CRITICAL,
        TRANSPORT_TIMEOUT_SATURATE,
    )


def _map_transport_rpc_health(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    service = str(up.get("service") or "unknown")
    timeouts = _int(up.get("timeout_count")) or 0
    successes = _int(up.get("success_count")) or 0
    p95 = _num(up.get("success_latency_ms_p95"))
    p50 = _num(up.get("success_latency_ms_p50"))
    threshold = _num(up.get("latency_p95_threshold_ms"))
    timeout_elapsed = _num(up.get("timeout_elapsed_ms_max"))
    channels = up.get("channel_counts") if isinstance(up.get("channel_counts"), dict) else {}

    if p95 is None and timeouts <= 0:
        return _no_evidence("transport", reason, "rpc_health_snapshot without p95 or timeouts")

    t_band = _timeout_band(timeouts)
    ratio = (p95 / threshold) if (p95 is not None and threshold and threshold > 0) else 0.0
    l_band = banded(
        ratio,
        TRANSPORT_LATENCY_RATIO_DEGRADED,
        TRANSPORT_LATENCY_RATIO_CRITICAL,
        TRANSPORT_LATENCY_RATIO_SATURATE,
    )
    thin = successes < TRANSPORT_MIN_CALLS_FOR_CRITICAL_LATENCY
    if thin:
        l_band = _cap(*l_band, "degraded")
    severity, magnitude = _worst(t_band, l_band)

    evidence: list[str] = []
    headline_bits: list[str] = []
    if timeouts > 0:
        total = timeouts + successes
        s = f"{service}: timeouts {timeouts}/{total} calls"
        if timeout_elapsed is not None:
            s += f" (longest wait {_secs(timeout_elapsed)})"
        evidence.append(s)
        headline_bits.append(f"{timeouts} RPC timeout(s)")
    if p95 is not None:
        s = f"{service}: p95 {_secs(p95)}"
        if threshold:
            s += f" vs limit {_secs(threshold)} ({ratio:.1f}x)"
        s += f", n={successes}"
        if thin:
            s += " (thin sample: p95 is the slowest call)"
        evidence.append(s)
        if p50 is not None:
            evidence.append(f"{service}: p50 {_secs(p50)}")
        if threshold:
            headline_bits.append(f"p95 {_secs(p95)} = {ratio:.1f}x limit on {successes} call(s)")
    if channels:
        mix = ", ".join(f"{k}={v}" for k, v in sorted(channels.items(), key=lambda kv: -(_num(kv[1]) or 0)))
        evidence.append(f"channels in window: {mix}")

    basis = f"timeouts={timeouts}, p95_ratio={ratio:.2f}, successes={successes}"
    return _build(
        kind="transport",
        severity=severity,
        magnitude=magnitude,
        basis=basis,
        evidence=evidence,
        touches=[service, *sorted(str(k) for k in channels)],
        headline=f"{service}: " + "; ".join(headline_bits or ["transport window flagged"]),
    )


def _map_transport_rpc_timeout(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    channel = str(up.get("request_channel") or "").strip()
    summary = str(up.get("summary") or "").strip()
    if not channel and not summary:
        return _no_evidence("transport", reason, "rpc_timeout without channel")
    severity, magnitude = _timeout_band(1)
    evidence = [f"RPC timeout on {channel or 'unknown channel'}"]
    m = _RPC_TIMEOUT_ELAPSED.search(summary)
    if m:
        evidence.append(f"waited {m.group(1)}s before giving up")
    return _build(
        kind="transport",
        severity=severity,
        magnitude=magnitude,
        basis="single rpc_transport_timeout grammar atom",
        evidence=evidence,
        touches=[channel] if channel else [],
        headline=f"RPC timeout on {channel or 'unknown channel'}",
    )


def _map_transport_bus_synaptic(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    err = _num(up.get("error"))
    thr = _num(up.get("error_threshold"))
    if err is None or not thr or thr <= 0:
        return _no_evidence("transport", reason, "bus_synaptic without error/threshold")
    ratio = err / thr
    severity, magnitude = banded(
        ratio, BUS_SYNAPTIC_RATIO_DEGRADED, BUS_SYNAPTIC_RATIO_CRITICAL, BUS_SYNAPTIC_RATIO_SATURATE
    )
    evidence = [f"bus prediction error {err:.3f} vs threshold {thr:.3f} ({ratio:.1f}x)"]
    if up.get("transition"):
        evidence.append(f"transition {up.get('transition')}")
    edges = _int(up.get("edge_count"))
    if edges is not None:
        evidence.append(f"edges measured: {edges}")
    return _build(
        kind="transport",
        severity=severity,
        magnitude=magnitude,
        basis=f"bus_synaptic error_ratio={ratio:.2f}",
        evidence=evidence,
        touches=["bus_mirror"],
        headline=f"bus inter-arrival prediction error {ratio:.1f}x its threshold",
    )


def _map_transport_baseline(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    """New condition-shaped upstream from the per-hop EWMA gate (spec A4)."""
    condition = str(up.get("condition") or "").strip().lower()
    phase = str(up.get("phase") or "open").strip().lower()
    key = str(up.get("key") or "").strip()
    if condition not in {"timeout", "spike", "saturation", "regime_shift", "zero_success"} or not key:
        return _no_evidence("transport", reason, f"transport_baseline condition={condition or 'missing'}")

    z = _num(up.get("z"))
    sat = _num(up.get("saturation_ratio"))
    baseline_ms = _num(up.get("baseline_ms"))
    floor_ms = _num(up.get("floor_ms"))
    window_ms = _num(up.get("window_mean_ms"))
    cpm = _num(up.get("calls_per_min"))
    cpm_usual = _num(up.get("calls_per_min_usual"))
    duration_s = _num(up.get("duration_s"))
    peak_ms = _num(up.get("peak_ms"))
    timeouts = _int(up.get("timeout_count")) or 0

    if condition == "zero_success":
        band: tuple[Severity, float] = ("critical", max(_timeout_band(timeouts)[1], CRITICAL_FLOOR))
        basis = f"zero_success, timeouts={timeouts}"
    elif condition == "timeout":
        band = _timeout_band(max(timeouts, 1))
        basis = f"timeouts={timeouts}"
    elif condition == "spike":
        if z is None:
            return _no_evidence("transport", reason, "spike without z")
        band = banded(z, TRANSPORT_Z_DEGRADED, TRANSPORT_Z_CRITICAL, TRANSPORT_Z_SATURATE)
        basis = f"z={z:.2f}"
    elif condition == "saturation":
        if sat is None:
            return _no_evidence("transport", reason, "saturation without saturation_ratio")
        band = banded(
            sat, TRANSPORT_SATURATION_DEGRADED, TRANSPORT_SATURATION_CRITICAL, TRANSPORT_SATURATION_SATURATE
        )
        basis = f"saturation_ratio={sat:.2f}"
    else:  # regime_shift: always stated, degraded (spec B table)
        if sat is not None:
            _, m = banded(
                sat, TRANSPORT_SATURATION_DEGRADED, TRANSPORT_SATURATION_CRITICAL, TRANSPORT_SATURATION_SATURATE
            )
            m = min(max(m, DEGRADED_FLOOR), CRITICAL_FLOOR - 0.0001)
        else:
            m = (DEGRADED_FLOOR + CRITICAL_FLOOR) / 2
        band = ("degraded", round(m, 4))
        basis = f"regime_shift saturation_ratio={sat if sat is not None else 'n/a'}"

    severity, magnitude = band
    if phase == "close":
        # An episode ending is news, not trouble: nominal, scaled by how big
        # the episode was so a big close still outranks a small one.
        severity, magnitude = "nominal", _nominal_band(magnitude)
        basis += ", phase=close"
    elif phase == "escalate":
        basis += ", phase=escalate"

    evidence: list[str] = [f"{condition} {phase} on {key}"]
    if window_ms is not None and baseline_ms is not None:
        s = f"mean {_secs(window_ms)} vs normal {_secs(baseline_ms)}"
        if z is not None:
            s += f" (z={z:.1f})"
        evidence.append(s)
    elif z is not None:
        evidence.append(f"z={z:.1f}")
    if sat is not None:
        s = f"saturation {sat:.2f}x best-recent normal"
        if floor_ms is not None:
            s += f" ({_secs(floor_ms)})"
        evidence.append(s)
    if timeouts:
        evidence.append(f"timeouts {timeouts}")
    if cpm is not None:
        busy = f"load {cpm:.1f}/min"
        if cpm_usual is not None:
            busy += f" vs usual {cpm_usual:.1f}/min"
            busy += " (slow while busy)" if cpm > cpm_usual * 1.5 else (
                " (slow while idle)" if cpm < cpm_usual * 0.5 else ""
            )
        evidence.append(busy)
    if duration_s is not None:
        evidence.append(f"episode duration {duration_s:.0f}s")
    if peak_ms is not None:
        evidence.append(f"peak {_secs(peak_ms)}")

    return _build(
        kind="transport",
        severity=severity,
        magnitude=magnitude,
        basis=basis,
        evidence=evidence,
        touches=[key],
        headline=f"{condition} {phase} on {key}",
    )


def map_transport(reason: str, upstream: dict[str, Any]) -> EvidenceMapping:
    source = str(upstream.get("evidence_source") or "").strip()
    if source == "transport_baseline" or "condition" in upstream:
        return _map_transport_baseline(reason, upstream)
    if source == "rpc_health_snapshot":
        return _map_transport_rpc_health(reason, upstream)
    if source == "rpc_transport_timeout_grammar":
        return _map_transport_rpc_timeout(reason, upstream)
    if source == "bus_synaptic_prediction_error":
        return _map_transport_bus_synaptic(reason, upstream)
    return _no_evidence("transport", reason, f"unknown evidence_source={source or 'missing'}")


# --------------------------------------------------------------------------
# telemetry_anomaly
# --------------------------------------------------------------------------

# recon_loss / threshold (spec B table). Gate fires at >= 1x; live 7d: 82% of
# rows are 1-2x.
TELEMETRY_RATIO_DEGRADED = 1.5
TELEMETRY_RATIO_CRITICAL = 2.5
TELEMETRY_RATIO_SATURATE = 5.0


def map_telemetry_anomaly(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    loss = _num(up.get("recon_loss"))
    thr = _num(up.get("threshold"))
    if loss is None or not thr or thr <= 0:
        return _no_evidence("telemetry_anomaly", reason, "missing recon_loss/threshold")
    ratio = loss / thr
    severity, magnitude = banded(ratio, TELEMETRY_RATIO_DEGRADED, TELEMETRY_RATIO_CRITICAL, TELEMETRY_RATIO_SATURATE)
    direction = str(up.get("deviation_direction") or "unknown")
    tops = [str(t) for t in (up.get("top_channels") or []) if isinstance(t, (str, int, float))]
    encoder = str(up.get("encoder_id") or "").strip()
    evidence = [f"recon_loss {loss:.4f} vs threshold {thr:.4f} ({ratio:.2f}x), {direction}"]
    evidence += [f"top channel {t}" for t in tops[:3]]
    msd = _num(up.get("mean_signed_deviation"))
    if msd is not None:
        evidence.append(f"mean signed deviation {msd:+.4f}")
    touches = ([encoder] if encoder else []) + [
        "channel:" + t.split("=", 1)[0] for t in tops[:3] if t.split("=", 1)[0]
    ]
    lead = tops[0].split("=", 1)[0] if tops else "unknown channel"
    return _build(
        kind="telemetry_anomaly",
        severity=severity,
        magnitude=magnitude,
        basis=f"recon_ratio={ratio:.3f}",
        evidence=evidence,
        touches=touches,
        headline=f"mood-arc reconstruction {ratio:.2f}x its threshold ({direction}), led by {lead}",
    )


# --------------------------------------------------------------------------
# chat_turn
# --------------------------------------------------------------------------

_CHAT_CRITICAL_STEP = 0.1
_CHAT_DEGRADED_STEP = 0.1


def map_chat_turn(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    fired = [str(c) for c in (up.get("fired_conditions") or []) if c]
    compliance = up.get("compliance_verdict")
    alignment = up.get("alignment_verdict")
    timed_out = up.get("timed_out") is True
    exit_code = up.get("exit_code")
    disposition = up.get("disposition")
    strain = up.get("strain_unresolved") is True
    surprise = _num(up.get("surprise_level"))
    finalize_degraded = up.get("finalize_degraded_reason")
    boundary = up.get("boundary_register") is True

    critical: list[str] = []
    degraded: list[str] = []
    informational: list[str] = []

    # spec B: failed compliance, timeout, or exit != 0 is critical.
    if compliance == "failed":
        critical.append("compliance failed")
    if timed_out:
        critical.append(f"turn timed out ({up.get('timeout_reason') or 'unknown'})")
    if exit_code not in (0, None) and _int(exit_code) not in (0, None):
        critical.append(f"exit_code={exit_code}")
    # misalignment or strain alone is degraded.
    if alignment == "misaligned":
        degraded.append("alignment misaligned")
    if strain:
        degraded.append("strain unresolved")
    if compliance == "partial":
        degraded.append("compliance partial")
    if disposition and disposition != "proceed":
        degraded.append(f"disposition={disposition}")
    if finalize_degraded:
        degraded.append(f"finalize degraded: {finalize_degraded}")
    # Informational: not a failure on its own.
    if alignment == "uncertain":
        informational.append("alignment uncertain")
    if boundary:
        informational.append("boundary register")
    if surprise is not None and any(c.startswith("surprise_level") for c in fired):
        informational.append(f"surprise {surprise:.2f}")

    if not (critical or degraded or informational):
        if not fired:
            return _no_evidence("chat_turn", reason, "no fired_conditions")
        informational.extend(fired)

    if critical:
        severity: Severity = "critical"
        magnitude = CRITICAL_FLOOR + _CHAT_CRITICAL_STEP * (len(critical) - 1) + 0.05 * len(degraded)
        magnitude = min(1.0, magnitude)
    elif degraded:
        severity = "degraded"
        magnitude = min(CRITICAL_FLOOR - 0.0001, DEGRADED_FLOOR + _CHAT_DEGRADED_STEP * (len(degraded) - 1))
    else:
        severity = "nominal"
        magnitude = min(DEGRADED_FLOOR - 0.0001, 0.1 * len(informational))

    evidence = critical + degraded + informational
    grounding = up.get("grounding_status")
    if grounding:
        evidence.append(f"grounding: {grounding}")
    for note in (up.get("alignment_notes") or [])[:2]:
        evidence.append(f"note: {note}")

    touches = ["chat_turn"]
    if compliance is not None or exit_code is not None:
        touches.append("harness_run")
    if alignment is not None or up.get("strain_unresolved") is not None:
        touches.append("reflection")
    if surprise is not None:
        touches.append("substrate_appraisal")
    if disposition is not None or boundary:
        touches.append("thought_event")

    return _build(
        kind="chat_turn",
        severity=severity,
        magnitude=magnitude,
        basis=f"critical={len(critical)}, degraded={len(degraded)}, informational={len(informational)}",
        evidence=evidence,
        touches=touches,
        headline="; ".join((critical + degraded + informational)[:3]),
    )


# --------------------------------------------------------------------------
# relational / repair_pressure_trend
# --------------------------------------------------------------------------

# repair_pressure_v2 level. Gate floor is 0.5 (EQUILIBRIUM_METACOG_RELATIONAL_LEVEL_THRESHOLD).
RELATIONAL_LEVEL_DEGRADED = 0.6
RELATIONAL_LEVEL_CRITICAL = 0.8
RELATIONAL_LEVEL_SATURATE = 1.0

# repair_pressure trend z. Gate fires at z >= 1.0 sustained 3 ticks; live
# z is ~1.0-1.2, i.e. a weak lift.
REPAIR_TREND_Z_DEGRADED = 2.0
REPAIR_TREND_Z_CRITICAL = 3.0
REPAIR_TREND_Z_SATURATE = 5.0


def map_relational(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    level = _num(up.get("level"))
    conf = _num(up.get("confidence"))
    if level is None:
        return _no_evidence("relational", reason, "missing level")
    severity, magnitude = banded(level, RELATIONAL_LEVEL_DEGRADED, RELATIONAL_LEVEL_CRITICAL, RELATIONAL_LEVEL_SATURATE)
    evidence = [
        f"repair pressure {level:.2f} ({up.get('level_label') or '?'})"
        + (f", confidence {conf:.2f}" if conf is not None else "")
    ]
    kinds: list[str] = []
    for e in up.get("evidence") or []:
        if not isinstance(e, dict):
            continue
        score = _num(e.get("score"))
        k = e.get("evidence_kind")
        if k and score is not None and score >= 0.5:
            kinds.append(str(k))
    if kinds:
        evidence.append("strong cues: " + ", ".join(kinds))
    if up.get("behavior_applied"):
        evidence.append(f"behavior applied: {up.get('behavior_applied')}")
    return _build(
        kind="relational",
        severity=severity,
        magnitude=magnitude,
        basis=f"level={level:.3f}",
        evidence=evidence,
        touches=["repair_pressure", *[f"repair:{k}" for k in kinds]],
        headline=f"repair pressure {level:.2f}" + (f" ({', '.join(kinds[:3])})" if kinds else ""),
    )


def map_repair_pressure_trend(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    z = _num(up.get("latest_zscore"))
    if z is None:
        return _no_evidence("repair_pressure_trend", reason, "missing latest_zscore")
    severity, magnitude = banded(z, REPAIR_TREND_Z_DEGRADED, REPAIR_TREND_Z_CRITICAL, REPAIR_TREND_Z_SATURATE)
    level = _num(up.get("latest_level"))
    base = _num(up.get("baseline_ewma"))
    consecutive = _int(up.get("consecutive_elevated"))
    evidence = [f"repair pressure z={z:.2f}" + (f" over {consecutive} consecutive readings" if consecutive else "")]
    if level is not None and base is not None:
        evidence.append(f"level {level:.2f} vs baseline {base:.2f}")
    return _build(
        kind="repair_pressure_trend",
        severity=severity,
        magnitude=magnitude,
        basis=f"z={z:.3f}",
        evidence=evidence,
        touches=["repair_pressure"],
        headline=f"repair pressure trending up (z={z:.2f})",
    )


# --------------------------------------------------------------------------
# insight / flow (attention self-model confidence patterns: news, not trouble)
# --------------------------------------------------------------------------


def map_insight(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    low = _num(up.get("low_value"))
    high = _num(up.get("high_value"))
    if low is None or high is None:
        return _no_evidence("insight", reason, "missing low_value/high_value")
    jump = max(0.0, high - low)
    ticks = _int(up.get("ticks_to_cross"))
    span = _num(up.get("cross_span_sec"))
    evidence = [f"confidence recovered {low:.3f} -> {high:.3f} (+{jump:.3f})"]
    if ticks is not None:
        evidence.append(f"crossed in {ticks} tick(s)" + (f" / {span:.0f}s" if span is not None else ""))
    source = str(up.get("evidence_source") or "attention_self_model")
    return _build(
        kind="insight",
        severity="nominal",
        magnitude=_nominal_band(jump),
        basis=f"confidence_jump={jump:.3f}",
        evidence=evidence,
        touches=[source],
        headline=f"prediction confidence recovered {low:.2f} -> {high:.2f}",
    )


def map_flow(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    mean = _num(up.get("mean_value"))
    floor = _num(up.get("floor"))
    if mean is None or floor is None or floor >= 1.0:
        return _no_evidence("flow", reason, "missing mean_value/floor")
    height = _clip01((mean - floor) / (1.0 - floor))
    stdev = _num(up.get("stdev_value"))
    span = _num(up.get("span_sec"))
    ticks = _int(up.get("tick_count"))
    evidence = [f"confidence held mean {mean:.3f} above floor {floor:.2f}"]
    if stdev is not None:
        evidence.append(f"stdev {stdev:.4f}")
    if span is not None:
        evidence.append(f"sustained {span:.0f}s" + (f" over {ticks} ticks" if ticks else ""))
    source = str(up.get("evidence_source") or "attention_self_model")
    return _build(
        kind="flow",
        severity="nominal",
        magnitude=_nominal_band(height),
        basis=f"height_above_floor={height:.3f}",
        evidence=evidence,
        touches=[source],
        headline=f"steady high prediction confidence (mean {mean:.2f}) for {span or 0:.0f}s",
    )


# --------------------------------------------------------------------------
# llm_surface_instability / baseline / manual
# --------------------------------------------------------------------------

LLM_UNSTABLE_SPANS_DEGRADED = 2.0
LLM_UNSTABLE_SPANS_CRITICAL = 3.0
LLM_UNSTABLE_SPANS_SATURATE = 6.0


def map_llm_surface_instability(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    unc = up.get("llm_uncertainty") if isinstance(up.get("llm_uncertainty"), dict) else {}
    spans = _num(unc.get("unstable_span_count"))
    if spans is None:
        return _no_evidence("llm_surface_instability", reason, "missing unstable_span_count")
    severity, magnitude = banded(
        spans, LLM_UNSTABLE_SPANS_DEGRADED, LLM_UNSTABLE_SPANS_CRITICAL, LLM_UNSTABLE_SPANS_SATURATE
    )
    low_margin = _int(unc.get("low_margin_token_count"))
    tokens = _int(unc.get("token_count_observed"))
    phase = str(up.get("phase") or "unknown")
    evidence = [f"{int(spans)} unstable span(s) in {phase}"]
    if low_margin is not None and tokens:
        evidence.append(f"low-margin tokens {low_margin}/{tokens}")
    return _build(
        kind="llm_surface_instability",
        severity=severity,
        magnitude=magnitude,
        basis=f"unstable_spans={spans:.0f}",
        evidence=evidence,
        touches=["orion-mind", f"phase:{phase}"],
        headline=f"{int(spans)} unstable language span(s) during {phase}",
    )


def map_baseline(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    return EvidenceMapping(
        severity="nominal",
        magnitude=0.0,
        density_label="ambient",
        density_rationale="event_magnitude[baseline]: scheduled check carries no event",
        evidence=_clean([f"reason={reason}" if reason else "scheduled_check"]),
        touches=[],
        summary_fallback="baseline (nominal): scheduled check, no event fired.",
    )


def map_manual(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    event_id = str(up.get("event_id") or "").strip()
    return EvidenceMapping(
        severity="nominal",
        magnitude=0.0,
        density_label="ambient",
        density_rationale="event_magnitude[manual]: operator-initiated, no measured magnitude",
        evidence=_clean([f"reason={reason}" if reason else "", f"source event {event_id}" if event_id else ""]),
        touches=["collapse_mirror"] if event_id else [],
        summary_fallback=f"manual (nominal): {reason or 'operator-initiated entry'}."[:300],
    )


def map_substrate(reason: str, up: dict[str, Any]) -> EvidenceMapping:
    """dense / pulse (substrate_metacog_gate.py). Never fired in the live
    table as of 2026-09-24, but the producer exists, so it is mapped rather
    than left to fall through to no_evidence. substrate_score is already a
    0..1 eventfulness score; it is used as the magnitude directly."""
    score = _num(up.get("substrate_score"))
    if score is None:
        return _no_evidence("substrate", reason, "missing substrate_score")
    severity, magnitude = banded(_clip01(score), DEGRADED_FLOOR, CRITICAL_FLOOR, 1.0)
    reasons = [str(r) for r in (up.get("reasons") or []) if r][:4]
    evidence = [f"substrate eventfulness {score:.2f}"] + [f"reason {r}" for r in reasons]
    return _build(
        kind="substrate",
        severity=severity,
        magnitude=magnitude,
        basis=f"substrate_score={score:.3f}",
        evidence=evidence,
        touches=["execution_trajectory"],
        headline=f"execution trajectory eventfulness {score:.2f}" + (f" ({', '.join(reasons)})" if reasons else ""),
    )


_MAPPERS: dict[str, Callable[[str, dict[str, Any]], EvidenceMapping]] = {
    "dense": map_substrate,
    "pulse": map_substrate,
    "transport": map_transport,
    "telemetry_anomaly": map_telemetry_anomaly,
    "chat_turn": map_chat_turn,
    "relational": map_relational,
    "repair_pressure_trend": map_repair_pressure_trend,
    "insight": map_insight,
    "flow": map_flow,
    "baseline": map_baseline,
    "llm_surface_instability": map_llm_surface_instability,
    "manual": map_manual,
}

KNOWN_KINDS = frozenset(_MAPPERS)


def map_trigger(trigger_kind: Any, reason: Any, upstream: Any) -> EvidenceMapping:
    """Entry point. Never raises: any malformed input -> nominal/no_evidence."""
    kind = str(trigger_kind or "").strip().lower()
    reason_s = str(reason or "")
    mapper = _MAPPERS.get(kind)
    if mapper is None:
        return _no_evidence(kind or "unknown", reason_s, "unknown trigger_kind")
    if kind not in ("baseline", "manual") and not isinstance(upstream, dict):
        return _no_evidence(kind, reason_s, "upstream is not an object")
    try:
        return mapper(reason_s, upstream if isinstance(upstream, dict) else {})
    except Exception as exc:  # pragma: no cover - defensive; tests cover known shapes
        return _no_evidence(kind, reason_s, f"mapper_error:{type(exc).__name__}")
