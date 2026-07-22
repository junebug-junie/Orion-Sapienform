from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from app.telemetry_anomaly_metacog_gate import build_telemetry_anomaly_metacog_trigger


def _score(
    *,
    recon_loss=0.01,
    recon_error_p95=0.0014,
    anomalous=True,
    top_channels=None,
    deviation_direction="elevated",
    mean_signed_deviation=0.05,
) -> dict:
    return {
        "correlation_id": "corr-score-1",
        "encoder_id": "mood-arc-encoder:field_channel_anomaly.v1",
        "encoder_version": "field_channel_anomaly.v1",
        "recon_loss": recon_loss,
        "recon_error_p95": recon_error_p95,
        "threshold_multiplier": 3.0,
        "threshold": recon_error_p95 * 3.0,
        "anomalous": anomalous,
        "window_start": "2026-07-21T04:00:00+00:00",
        "window_end": "2026-07-21T04:01:00+00:00",
        "window_size": 30,
        "top_channels": top_channels if top_channels is not None else ["cpu_pressure=0.01234", "transport_pressure=0.00456"],
        "deviation_direction": deviation_direction,
        "mean_signed_deviation": mean_signed_deviation,
    }


def test_recon_loss_above_own_threshold_fires_telemetry_anomaly_trigger():
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-1",
        score=_score(recon_loss=0.01, recon_error_p95=0.0014),
        zen_state="not_zen",
        pressure=0.4,
        recall_enabled=False,
        threshold_multiplier=3.0,
    )
    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "telemetry_anomaly"
    assert trigger.signal_refs == ["corr-1"]
    assert trigger.upstream["recon_loss"] == 0.01
    assert trigger.upstream["recon_error_p95"] == 0.0014
    assert trigger.upstream["threshold"] == 0.0014 * 3.0
    assert trigger.upstream["encoder_version"] == "field_channel_anomaly.v1"
    assert trigger.upstream["top_channels"] == ["cpu_pressure=0.01234", "transport_pressure=0.00456"]
    assert trigger.reason.endswith(":top=cpu_pressure=0.01234")
    assert trigger.upstream["deviation_direction"] == "elevated"
    assert trigger.upstream["mean_signed_deviation"] == 0.05
    assert ":elevated:" in trigger.reason


def test_depressed_direction_flows_into_reason_and_upstream():
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-8",
        score=_score(deviation_direction="depressed", mean_signed_deviation=-0.05),
        zen_state="not_zen",
        pressure=0.4,
        recall_enabled=False,
        threshold_multiplier=3.0,
    )
    assert trigger.upstream["deviation_direction"] == "depressed"
    assert ":depressed:" in trigger.reason


def test_missing_deviation_direction_defaults_to_mixed_not_a_crash():
    score = _score()
    del score["deviation_direction"]
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-9",
        score=score,
        zen_state="not_zen",
        pressure=0.4,
        recall_enabled=False,
        threshold_multiplier=3.0,
    )
    assert trigger.upstream["deviation_direction"] == "mixed"
    assert ":mixed:" in trigger.reason


def test_missing_top_channels_does_not_break_trigger_construction():
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-7",
        score=_score(recon_loss=0.01, recon_error_p95=0.0014, top_channels=[]),
        zen_state="not_zen",
        pressure=0.4,
        recall_enabled=False,
        threshold_multiplier=3.0,
    )
    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.upstream["top_channels"] == []
    assert ":top=" not in trigger.reason


def test_recon_loss_at_or_below_threshold_does_not_fire():
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-2",
        score=_score(recon_loss=0.002, recon_error_p95=0.0014),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        threshold_multiplier=3.0,
    )
    assert trigger is None


def test_consumer_threshold_multiplier_overrides_producer_anomalous_flag():
    """The producer's own `anomalous: True` must not matter -- this
    service applies its own multiplier, same principle as
    build_repair_pressure_metacog_trigger's floors."""
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-3",
        score=_score(recon_loss=0.003, recon_error_p95=0.0014, anomalous=True),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        threshold_multiplier=10.0,
    )
    assert trigger is None


def test_missing_score_does_not_fire():
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-4",
        score=None,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        threshold_multiplier=3.0,
    )
    assert trigger is None


def test_malformed_score_missing_recon_loss_does_not_fire():
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-5",
        score={"recon_error_p95": 0.0014},
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        threshold_multiplier=3.0,
    )
    assert trigger is None


def test_exactly_at_threshold_does_not_fire():
    """assert-strict '>' (not '>='), same boundary convention as
    build_repair_pressure_metacog_trigger's '<' floor check."""
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id="corr-6",
        score=_score(recon_loss=0.0042, recon_error_p95=0.0014),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        threshold_multiplier=3.0,
    )
    assert trigger is None
