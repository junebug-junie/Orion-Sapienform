from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_list(val: str | None) -> List[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    service_name: str = Field("orion-equilibrium-service", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: Optional[str] = Field(None, alias="NODE_NAME")
    instance_id: Optional[str] = Field(None, alias="INSTANCE_ID")

    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")

    expected_services_raw: Optional[str] = Field(None, alias="EQUILIBRIUM_EXPECTED_SERVICES")
    expected_services_path: Optional[Path] = Field(None, alias="EQUILIBRIUM_EXPECTED_SERVICES_PATH")
    grace_multiplier: float = Field(3.0, alias="EQUILIBRIUM_GRACE_MULTIPLIER")
    publish_interval_sec: float = Field(15.0, alias="EQUILIBRIUM_PUBLISH_INTERVAL_SEC")
    windows_sec: List[int] = Field(default_factory=lambda: [60, 300, 3600], alias="EQUILIBRIUM_WINDOWS_SEC")
    collapse_mirror_interval_sec: float = Field(15.0, alias="EQUILIBRIUM_COLLAPSE_MIRROR_INTERVAL_SEC")

    redis_state_key: str = Field("equilibrium:state", alias="EQUILIBRIUM_STATE_KEY")
    channel_equilibrium_snapshot: str = Field("orion:equilibrium:snapshot", alias="CHANNEL_EQUILIBRIUM_SNAPSHOT")
    channel_spark_signal: str = Field("orion:spark:signal", alias="CHANNEL_SPARK_SIGNAL")
    # Historical downtime persistence (2026-07-24, docs/superpowers/specs/
    # 2026-07-24-service-heartbeat-node-telemetry-design.md). A separate
    # channel from CHANNEL_EQUILIBRIUM_SNAPSHOT on purpose -- see that
    # channel's own registry comment in orion/bus/channels.yaml for why
    # sql-writer must not subscribe to the periodic snapshot channel
    # directly. Enabled by default (this is the feature Juniper asked to
    # build); the flag exists as a cheap kill switch, not a staged rollout.
    channel_equilibrium_transition: str = Field(
        "orion:equilibrium:transition", alias="CHANNEL_EQUILIBRIUM_TRANSITION"
    )
    equilibrium_transition_publish_enable: bool = Field(
        True, alias="EQUILIBRIUM_TRANSITION_PUBLISH_ENABLE"
    )
    state_retention_sec: float = Field(3600.0, alias="EQUILIBRIUM_STATE_RETENTION_SEC")
    channel_metacognition_tick: str = Field("orion:metacognition:tick", alias="CHANNEL_METACOGNITION_TICK")
    channel_cognition_trace_pub: str = Field("orion:cognition:trace", alias="CHANNEL_COGNITION_TRACE_PUB")

    equilibrium_spark_heartbeat_enable: bool = Field(False, alias="EQUILIBRIUM_SPARK_HEARTBEAT_ENABLE")
    equilibrium_spark_heartbeat_interval_sec: float = Field(
        30.0, alias="EQUILIBRIUM_SPARK_HEARTBEAT_INTERVAL_SEC"
    )

    # Metacognition
    metacog_enable: bool = Field(False, alias="EQUILIBRIUM_METACOG_ENABLE")
    metacog_baseline_interval_sec: float = Field(3600.0, alias="EQUILIBRIUM_METACOG_BASELINE_INTERVAL_SEC")
    metacog_baseline_max_skips: int = Field(3, alias="EQUILIBRIUM_METACOG_BASELINE_MAX_SKIPS")
    metacog_cooldown_sec: float = Field(30.0, alias="EQUILIBRIUM_METACOG_COOLDOWN_SEC")
    metacog_substrate_trigger_enable: bool = Field(
        True, alias="EQUILIBRIUM_METACOG_SUBSTRATE_TRIGGER_ENABLE"
    )
    metacog_substrate_dense_threshold: float = Field(
        0.55, alias="EQUILIBRIUM_METACOG_SUBSTRATE_DENSE_THRESHOLD"
    )
    metacog_substrate_pulse_threshold: float = Field(
        0.30, alias="EQUILIBRIUM_METACOG_SUBSTRATE_PULSE_THRESHOLD"
    )
    metacog_recall_enabled: bool = Field(
        False,
        alias="EQUILIBRIUM_METACOG_RECALL_ENABLED",
    )
    # Relational metacog trigger. Same concept as before (a real, live signal about
    # something notable in the conversation/relationship, distinct from execution
    # mechanics) but a different source: repair_pressure_v2 (orion:repair_pressure:appraisal,
    # published by orion-hub whenever the repair_pressure paradigm actually runs) replaced
    # the retired orion/memory/turn_change_classify.py SHIFT gate. Names kept on purpose.
    metacog_relational_trigger_enable: bool = Field(
        True, alias="EQUILIBRIUM_METACOG_RELATIONAL_TRIGGER_ENABLE"
    )
    metacog_relational_confidence_threshold: float = Field(
        0.7, alias="EQUILIBRIUM_METACOG_RELATIONAL_CONFIDENCE_THRESHOLD"
    )
    metacog_relational_level_threshold: float = Field(
        0.5, alias="EQUILIBRIUM_METACOG_RELATIONAL_LEVEL_THRESHOLD"
    )

    # Telemetry-anomaly metacog trigger (2026-07-21). Mirrors the relational
    # trigger's shape: orion-field-digester publishes the raw measurement
    # (recon_loss vs. its own train-time recon_error_p95), this service owns
    # the trigger-sensitivity knob via its own threshold multiplier rather
    # than trusting the producer's embedded "anomalous" flag.
    metacog_telemetry_anomaly_trigger_enable: bool = Field(
        True, alias="EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE"
    )
    # Matches fit_encoder.py detect-anomalies's own --threshold-multiplier
    # default (3.0x the encoder's held-out recon_error_p95).
    metacog_telemetry_anomaly_threshold_multiplier: float = Field(
        3.0, alias="EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_THRESHOLD_MULTIPLIER"
    )

    metacog_publish_verb_request: bool = Field(
        False,
        alias="EQUILIBRIUM_METACOG_PUBLISH_VERB_REQUEST",
    )

    # chat_turn metacog trigger (2026-07-22). Correlates ThoughtEventV1
    # (orion:thought:artifact) + HarnessRunV1 (orion:harness:run:artifact) +
    # an optional governor-timeout GrammarEventV1 (orion:grammar:event,
    # semantic_role == "exec_turn_timeout", Patch B / PR #1287) by
    # correlation_id and fires on real gate conditions read off those two
    # schemas. See chat_turn_metacog_gate.py and docs/superpowers/design/
    # 2026-07-18-collapse-mirror-metacog-redesign.md.
    metacog_chat_turn_trigger_enable: bool = Field(
        True, alias="EQUILIBRIUM_METACOG_CHAT_TURN_TRIGGER_ENABLE"
    )
    metacog_chat_turn_correlator_ttl_sec: int = Field(
        600, alias="EQUILIBRIUM_METACOG_CHAT_TURN_CORRELATOR_TTL_SEC"
    )
    metacog_chat_turn_surprise_threshold: float = Field(
        0.7, alias="EQUILIBRIUM_METACOG_CHAT_TURN_SURPRISE_THRESHOLD"
    )
    # Separate cooldown lane from EQUILIBRIUM_METACOG_COOLDOWN_SEC (2026-07-23).
    # chat_turn is the only trigger kind designed to fire on essentially every
    # remarkable chat turn rather than a rare/periodic cadence -- sharing the
    # global cooldown timestamp would let a burst of chat_turn fires silently
    # starve baseline/manual/pulse/relational/telemetry_anomaly's own fires,
    # not just drop chat_turn's own excess. Same default value as the shared
    # cooldown to start; independently tunable from here on.
    metacog_chat_turn_cooldown_sec: float = Field(
        30.0, alias="EQUILIBRIUM_METACOG_CHAT_TURN_COOLDOWN_SEC"
    )

    # transport metacog trigger (docs/superpowers/specs/2026-07-24-transport-metacog-
    # trigger-design.md). Own cooldown lane from day one -- chat_turn's own cooldown
    # bug (a shared lane let a burst of one kind starve the others) should not repeat.
    metacog_transport_trigger_enable: bool = Field(
        False, alias="EQUILIBRIUM_METACOG_TRANSPORT_TRIGGER_ENABLE"
    )
    metacog_transport_cooldown_sec: float = Field(
        30.0, alias="EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC"
    )
    # RpcHealthSnapshotV1 p95 latency threshold (Option A gate condition). No existing
    # config to inherit from -- a starting default, same calibration caveat as
    # telemetry_anomaly's threshold_multiplier; needs real data to validate.
    metacog_transport_latency_p95_threshold_ms: float = Field(
        5000.0, alias="EQUILIBRIUM_METACOG_TRANSPORT_LATENCY_P95_THRESHOLD_MS"
    )
    # Option (bus_synaptic): third transport evidence source, reads
    # node:substrate.bus_synaptic's prediction_error directly from FalkorDB
    # (orion_substrate graph, written by orion-substrate-runtime's
    # _bus_synaptic_tick -- PR #1377/#1380). Passively covers RPC-health-
    # invisible organs (bespoke long-poll clients like orion-harness-governor)
    # that Options A/C structurally cannot see -- see
    # docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md's
    # 2026-07-25 revisions. Shares trigger_kind="transport"'s existing cooldown
    # lane -- same kind, third evidence branch, not a new one.
    metacog_transport_bus_synaptic_poll_enable: bool = Field(
        False, alias="EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_POLL_ENABLE"
    )
    metacog_transport_bus_synaptic_poll_interval_sec: float = Field(
        30.0, alias="EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_POLL_INTERVAL_SEC"
    )
    # 1.0 = bus_synaptic_prediction_error's own saturation ceiling, which by
    # construction means the aggregated edges' mean |zscore| already reached
    # 3.0 -- reuses the same anomaly bar Hub's own debug routes use
    # (zscore_threshold=3.0), not a new arbitrary threshold.
    metacog_transport_bus_synaptic_error_threshold: float = Field(
        1.0, alias="EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_ERROR_THRESHOLD"
    )
    # ---------------------------------------------------------------------
    # Generative (non-rupture) metacog triggers: insight + flow.
    # docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md
    #
    # Both read one already-live field, AttentionSelfModelV1.prediction_error_
    # confidence, from `substrate_attention_self_model` (written every ~30s by
    # orion-substrate-runtime's _attention_self_model_tick, PR #1459) as two
    # different windowing functions -- insight looks for a low->high transition,
    # flow looks for a sustained plateau. No new producer, reducer or schema.
    #
    # Both ship DISABLED. Matching the bus_synaptic precedent (PR #1385 ->
    # #1387): a gate that dispatches a real MetacogTriggerV1 into orion_metacog
    # gets flipped on by a human only after its own post-merge live-data check,
    # separately from "is the underlying signal worth reading."
    # ---------------------------------------------------------------------
    metacog_insight_trigger_enable: bool = Field(
        False, alias="EQUILIBRIUM_METACOG_INSIGHT_TRIGGER_ENABLE"
    )
    metacog_flow_trigger_enable: bool = Field(
        False, alias="EQUILIBRIUM_METACOG_FLOW_TRIGGER_ENABLE"
    )
    # One poll serves both gates -- same table, same row window, evaluated twice.
    # Matches the ~30s cadence of the tick that writes the rows.
    metacog_generative_poll_interval_sec: float = Field(
        30.0, alias="EQUILIBRIUM_METACOG_GENERATIVE_POLL_INTERVAL_SEC"
    )
    metacog_generative_postgres_uri: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        alias="EQUILIBRIUM_METACOG_GENERATIVE_POSTGRES_URI",
    )
    # Own cooldown lanes, from day one. chat_turn shipped the shared-lane bug
    # once already (a burst of one kind silently starved every other kind);
    # see _PER_KIND_COOLDOWN_SETTINGS_ATTR in service.py. Generous defaults:
    # both of these describe slow-moving regimes, not per-turn events, so
    # re-announcing the same calm every 30s would be noise.
    metacog_insight_cooldown_sec: float = Field(
        300.0, alias="EQUILIBRIUM_METACOG_INSIGHT_COOLDOWN_SEC"
    )
    metacog_flow_cooldown_sec: float = Field(
        1800.0, alias="EQUILIBRIUM_METACOG_FLOW_COOLDOWN_SEC"
    )
    # PROVISIONAL thresholds, pending the longer-window re-run scheduled
    # 2026-08-02. Picked in PR #1463's baseline pass to match this metric's real
    # observed range in this environment (the design doc's original 0.5/0.8
    # anchors fired zero events because prediction_error_confidence never once
    # dropped to 0.5 in 14h); re-measured 2026-07-30 over 20.6h/2265 ticks:
    # range 0.597-0.977, mean 0.893, with 14 ticks at/below 0.70 and 1544
    # at/above 0.90 -- so both bands are real and reachable, not degenerate.
    metacog_insight_low_threshold: float = Field(
        0.70, alias="EQUILIBRIUM_METACOG_INSIGHT_LOW_THRESHOLD"
    )
    metacog_insight_high_threshold: float = Field(
        0.90, alias="EQUILIBRIUM_METACOG_INSIGHT_HIGH_THRESHOLD"
    )
    # Trailing rows fetched per poll. Must cover the widest window either
    # detector needs (insight's max_ticks_to_cross + confirm, flow's min_ticks).
    metacog_generative_window_ticks: int = Field(
        20, alias="EQUILIBRIUM_METACOG_GENERATIVE_WINDOW_TICKS"
    )
    # Observed ticks-to-cross were 1, 3 and 12 (PR #1463) -- 15 covers the
    # measured max with headroom while still rejecting a low from hours ago
    # being retroactively called a recovery. Provisional, same 2026-08-02 re-run.
    metacog_insight_max_ticks_to_cross: int = Field(
        15, alias="EQUILIBRIUM_METACOG_INSIGHT_MAX_TICKS_TO_CROSS"
    )
    # How many consecutive newest ticks must hold the high band before a climb
    # counts as resolved. 2 ticks (~60s) -- recoveries are a median 3-tick
    # gradual climb, so a single-tick check would fire partway up. 1229 real
    # consecutive >=0.90 pairs in the 20.6h window, so this is easily reachable.
    metacog_insight_confirm_ticks: int = Field(
        2, alias="EQUILIBRIUM_METACOG_INSIGHT_CONFIRM_TICKS"
    )
    # Flow: min over the window >= floor AND stdev <= max_stdev. Calibrated
    # 2026-07-30 against 2246 real 20-tick windows: at floor=0.90, 71 windows
    # (3.2%) qualify -- selective but non-degenerate. floor=0.92 measured 0
    # qualifying windows (degenerate, do not use). The variance ceiling is
    # currently non-binding at floor=0.90 (identical 71 windows at 0.02/0.03/
    # 0.05, since the field's ~0.977 ceiling squeezes any min>=0.90 window into
    # a <0.08 band) and is kept for when the floor is lowered or the field's
    # range shifts -- see detect_flow_regime's docstring. Provisional.
    metacog_flow_floor: float = Field(0.90, alias="EQUILIBRIUM_METACOG_FLOW_FLOOR")
    metacog_flow_max_stdev: float = Field(
        0.02, alias="EQUILIBRIUM_METACOG_FLOW_MAX_STDEV"
    )
    # ~10 minutes of consecutive calm at a ~30s tick cadence.
    metacog_flow_min_ticks: int = Field(20, alias="EQUILIBRIUM_METACOG_FLOW_MIN_TICKS")

    falkordb_uri: str = Field("redis://localhost:6379", alias="FALKORDB_URI")
    falkordb_substrate_graph: str = Field(
        "orion_substrate", alias="FALKORDB_SUBSTRATE_GRAPH"
    )

    channel_metacog_trigger: str = Field("orion:equilibrium:metacog:trigger", alias="CHANNEL_EQUILIBRIUM_METACOG_TRIGGER")
    channel_collapse_mirror_user_event: str = Field("orion:collapse:intake", alias="CHANNEL_COLLAPSE_MIRROR_USER_EVENT")
    channel_repair_pressure_appraisal: str = Field(
        "orion:repair_pressure:appraisal", alias="CHANNEL_REPAIR_PRESSURE_APPRAISAL"
    )
    channel_field_channel_anomaly_score: str = Field(
        "orion:field_channel:anomaly_score", alias="CHANNEL_FIELD_CHANNEL_ANOMALY_SCORE"
    )
    channel_thought_artifact: str = Field("orion:thought:artifact", alias="CHANNEL_THOUGHT_ARTIFACT")
    channel_harness_run_artifact: str = Field(
        "orion:harness:run:artifact", alias="CHANNEL_HARNESS_RUN_ARTIFACT"
    )
    channel_grammar_event: str = Field("orion:grammar:event", alias="CHANNEL_GRAMMAR_EVENT")
    channel_rpc_health_snapshot: str = Field(
        "orion:rpc_health:snapshot", alias="CHANNEL_RPC_HEALTH_SNAPSHOT"
    )

    @model_validator(mode="after")
    def _coerce_windows(self) -> "Settings":
        if isinstance(self.windows_sec, str):
            try:
                parsed = json.loads(self.windows_sec)
                if isinstance(parsed, list):
                    self.windows_sec = [int(x) for x in parsed]
            except Exception:
                self.windows_sec = [int(x) for x in _parse_list(self.windows_sec)]
        return self

    def expected_services(self) -> List[str]:
        items: List[str] = []
        items.extend(_parse_list(self.expected_services_raw))

        if self.expected_services_path and self.expected_services_path.exists():
            try:
                import yaml

                raw = yaml.safe_load(self.expected_services_path.read_text()) or {}
                from_file = raw.get("services") if isinstance(raw, dict) else raw
                if isinstance(from_file, list):
                    items.extend([str(x) for x in from_file])
            except Exception:
                pass

        # ensure uniqueness while preserving order
        deduped: List[str] = []
        for it in items:
            if it not in deduped:
                deduped.append(it)
        return deduped


settings = Settings()
