"""Thin helpers for populating GrammarAtomV1.uncertainty from existing signals.

Provenance is always an already-computed metric on the producer side — no new
taxonomy, no keyword lists. Heartbeat's 2-site entangling gate reads
``1.0 - uncertainty`` alongside ``confidence`` (see
services/orion-heartbeat/app/substrate/mps_state.py).
"""
from __future__ import annotations

from typing import Any

from orion.schemas.telemetry.biometrics import BiometricsInductionV1


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def uncertainty_from_telemetry_error_rate(error_rate: float) -> float:
    """BiometricsSummaryV1.telemetry_error_rate (orion/telemetry/biometrics_pipeline.py)."""
    return clamp01(error_rate)


def uncertainty_from_inverse_confidence(confidence: float) -> float:
    return clamp01(1.0 - confidence)


def uncertainty_from_induction_volatility(induction: BiometricsInductionV1) -> float:
    """Max metric.volatility across BiometricsInductionV1.metrics."""
    vols = [m.volatility for m in induction.metrics.values()]
    return clamp01(max(vols) if vols else 0.0)


def uncertainty_from_abs_zscore(zscore: float | None, *, no_baseline: float = 0.75) -> float:
    """Bus activity EWMA z-score; None when the tracker has no baseline yet."""
    if zscore is None:
        return clamp01(no_baseline)
    return clamp01(min(1.0, abs(float(zscore)) / 3.0))


def uncertainty_from_sample_mismatch(mismatch_count: int, sampled_count: int) -> float:
    """Schema-validation sample mismatch ratio (bus observer)."""
    if sampled_count <= 0:
        return 0.5
    return clamp01(mismatch_count / sampled_count)


def uncertainty_from_backpressure(stream_length: int, threshold: int) -> float:
    """How far stream depth exceeds the configured backpressure threshold."""
    if threshold <= 0:
        return 0.5
    if stream_length <= threshold:
        return clamp01(stream_length / threshold * 0.25)
    return clamp01((stream_length - threshold) / threshold)


def uncertainty_from_catalog_drift(undeclared_active_count: int, catalog_size: int) -> float:
    """Mesh-wide bus census undeclared-active fraction."""
    if catalog_size <= 0:
        return 0.5
    return clamp01(undeclared_active_count / catalog_size)


def uncertainty_from_route_arbitration(route_metadata: dict[str, Any]) -> float:
    """Categorical route facts from orchestrator.call_verb_runtime() route_metadata."""
    base = 0.12
    if not route_metadata.get("mind_requested"):
        reason = str(route_metadata.get("mind_skip_reason") or "")
        if reason and reason not in {"none", "unknown"}:
            base = max(base, 0.42)
    lane_reason = str(route_metadata.get("execution_lane_reason") or "")
    if lane_reason in {"unknown", "fallback", "lane_routing_disabled"}:
        base = max(base, 0.35)
    return clamp01(base)


def uncertainty_for_cortex_exec_atom(
    *,
    atom_type: str,
    semantic_role: str,
    confidence: float,
) -> float:
    """Epistemic slack for cortex-exec lifecycle atoms (no extra upstream metric)."""
    if atom_type == "uncertainty_marker":
        if semantic_role in {"exec_request_invalid", "exec_step_failed"}:
            return 0.85
        return 0.75
    if semantic_role == "exec_recall_gate_observed":
        return uncertainty_from_inverse_confidence(confidence)
    if semantic_role in {"exec_step_completed", "exec_result_assembled"}:
        return 0.18
    return uncertainty_from_inverse_confidence(confidence)
