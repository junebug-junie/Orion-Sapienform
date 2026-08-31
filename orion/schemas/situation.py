from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RequestorContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = "Juniper"
    relationship_to_orion: str = "primary_operator"
    source: str = "default"
    confidence: Literal["low", "medium", "high"] = "medium"


class PresenceCompanionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str
    relationship: Literal["spouse", "child", "friend", "coworker", "guest", "other"] = "other"
    role: Literal["listener", "asker", "participant", "nearby"] = "nearby"
    age_band: Literal["child", "teen", "adult", "unknown"] = "unknown"
    context_note: Optional[str] = None
    safety_notes: Optional[str] = None


class PresenceContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["presence.context.v1"] = "presence.context.v1"
    requestor: RequestorContextV1 = Field(default_factory=RequestorContextV1)
    companions: list[PresenceCompanionV1] = Field(default_factory=list)
    audience_mode: Literal[
        "solo",
        "family",
        "kid_present",
        "spouse_present",
        "mixed_group",
        "operator_review",
        "guest_present",
        "unknown",
    ] = "solo"
    submitted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    source: Literal["hub_manual", "payload", "default"] = "default"
    persist_to_memory: bool = False
    privacy_mode: Literal["session_only", "persist_allowed"] = "session_only"
    notes: Optional[str] = None


class TimeContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str = "America/Denver"
    local_datetime: str
    local_date: str
    local_time: str
    weekday: str
    is_weekend: bool
    season_local: str
    time_of_day_label: Literal[
        "pre_dawn",
        "early_morning",
        "mid_morning",
        "late_morning",
        "midday",
        "early_afternoon",
        "late_afternoon",
        "evening",
        "late_evening",
        "night",
    ]
    day_phase: Literal["pre_dawn", "dawn", "morning", "midday", "afternoon", "dusk", "night"]
    sun_phase: Literal["before_sunrise", "daylight", "after_sunset", "unknown"] = "unknown"
    sunrise_local: Optional[str] = None
    sunset_local: Optional[str] = None


class ConversationPhaseContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_user_turn_at: Optional[datetime] = None
    last_orion_turn_at: Optional[datetime] = None
    time_since_last_user_turn_seconds: Optional[int] = None
    time_since_last_orion_turn_seconds: Optional[int] = None
    crossed_day_boundary: bool = False
    phase_change: Literal[
        "same_breath",
        "short_pause",
        "resumed_thread",
        "long_gap",
        "next_day",
        "stale_thread",
        "unknown",
    ] = "unknown"
    continuity_mode: Literal["continue_directly", "lightly_resume", "reorient", "revalidate_context"] = "continue_directly"
    topic_staleness_risk: Literal["none", "low", "medium", "high"] = "none"
    response_adjustments: list[str] = Field(default_factory=list)


class PlaceContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coarse_location: str = "Unknown"
    locality: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    timezone: str = "America/Denver"
    precision: Literal["none", "coarse", "city", "exact"] = "coarse"
    source: Literal["configured_home", "browser_metadata", "manual", "unknown"] = "unknown"
    confidence: Literal["low", "medium", "high"] = "low"


class WeatherCurrentV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_f: Optional[float] = None
    feels_like_f: Optional[float] = None
    condition: str = "unknown"
    wind_mph: Optional[float] = None
    wind_gust_mph: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    pressure_trend: Optional[str] = None
    precipitation_now: Optional[str] = None
    visibility: Optional[str] = None
    air_quality: Optional[str] = None


class WeatherForecastWindowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_label: Literal["next_2h", "next_6h", "next_24h"]
    summary: str = "unknown"
    precipitation_probability_pct: Optional[int] = None
    precipitation_type: Optional[str] = None
    temperature_low_f: Optional[float] = None
    temperature_high_f: Optional[float] = None
    wind_max_mph: Optional[float] = None
    severe_risk: Optional[str] = None


class WeatherAlertV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    severity: Optional[str] = None
    starts_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None
    source: Optional[str] = None


class WeatherPracticalFlagsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    take_jacket: bool = False
    take_umbrella: bool = False
    high_wind: bool = False
    icy_roads: bool = False
    hot_car_risk: bool = False
    poor_air_quality: bool = False
    severe_weather: bool = False


class EnvironmentContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_weather: WeatherCurrentV1 = Field(default_factory=WeatherCurrentV1)
    forecast_next_2h: WeatherForecastWindowV1 = Field(
        default_factory=lambda: WeatherForecastWindowV1(window_label="next_2h")
    )
    forecast_next_6h: WeatherForecastWindowV1 = Field(
        default_factory=lambda: WeatherForecastWindowV1(window_label="next_6h")
    )
    forecast_next_24h: WeatherForecastWindowV1 = Field(
        default_factory=lambda: WeatherForecastWindowV1(window_label="next_24h")
    )
    weather_alerts: list[WeatherAlertV1] = Field(default_factory=list)
    practical_flags: WeatherPracticalFlagsV1 = Field(default_factory=WeatherPracticalFlagsV1)
    source_age_seconds: Optional[int] = None
    source: str = "none"
    confidence: Literal["low", "medium", "high"] = "low"
    available: bool = False


class AgendaContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = False
    next_event_title: Optional[str] = None
    next_event_start: Optional[datetime] = None
    minutes_until_next_event: Optional[int] = None
    has_school_day_context: Optional[bool] = None
    has_travel_today: Optional[bool] = None
    source: str = "none"
    confidence: Literal["low", "medium", "high"] = "low"
    diagnostics: dict[str, str] = Field(default_factory=dict)


class LabContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = False
    active_node: Optional[str] = None
    gpu_hosts_online: list[str] = Field(default_factory=list)
    service_health_summary: Optional[str] = None
    ambient_temp_f: Optional[float] = None
    power_load_watts: Optional[float] = None
    ups_on_battery: Optional[bool] = None
    thermal_risk: Literal["unknown", "low", "medium", "high"] = "unknown"
    power_risk: Literal["unknown", "low", "medium", "high"] = "unknown"
    source: str = "none"
    confidence: Literal["low", "medium", "high"] = "low"
    diagnostics: dict[str, str] = Field(default_factory=dict)


class CabinetContextV1(BaseModel):
    """Live read of Orion's own physical cabinet sensors (Athena Nano ESP32
    node: BME680 climate, LTR390 UV/ALS, magnetometer, PMSA003I particulate,
    VL53L1X lidar, BNO085 IMU/vibration -- see
    `orion.telemetry.cabinet_sensors`'s module docstring for the sensor
    inventory and `services/orion-biometrics/README.md`'s "Cabinet sensor
    node" section for the full host-reader -> biometrics pipeline).

    This is the real thing `LabContextV1` was a stand-in for -- Orion's own
    housing, not a hand-waved compute-cluster risk summary. `LabContextV1`
    stays untouched as a distinct, still-unwired concept (GPU-cluster
    thermal/power risk); this does not replace or extend it.

    Producer: `orion/situational/context.py`'s `_fetch_cabinet_context`,
    reusing the same shared `orion.telemetry.cabinet_sensors` /
    `cabinet_snapshot_merge` helpers `services/orion-hub/scripts/
    cabinet_sensors_routes.py`'s `/api/cabinet/sensors/latest` route already
    uses -- not an import of that route module itself (`orion/` is shared
    code services import FROM, never the reverse).

    Raw measurements are native units, present only when actually measured
    (absent, never a fabricated 0.0 -- same invariant
    `extract_cabinet_measurements` documents). The `*_activity` fields are
    baseline-relative 0-1 signals (EWMA band + volatility, HAND-VERIFIED to
    rest at exactly 0.0 for constant input -- see
    `orion.telemetry.cabinet_sensors`'s module docstring), not absolute
    comfort/AQI thresholds.

    `available=False` is a real state (disabled / no sensor-file mount
    configured / stale frame / empty measurement set / read failure) --
    same honesty contract as `PerceptionContextV1.available`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["cabinet.context.v1"] = "cabinet.context.v1"
    available: bool = False
    source: str = "none"
    age_seconds: Optional[float] = None
    device: Optional[str] = None
    temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    gas_resistance_ohm: Optional[float] = None
    uv_raw: Optional[float] = None
    als_raw: Optional[float] = None
    magnetic_ut: Optional[float] = None
    pm1_ug_m3: Optional[float] = None
    pm25_ug_m3: Optional[float] = None
    pm10_ug_m3: Optional[float] = None
    lidar_mm: Optional[float] = None
    vibration_g: Optional[float] = None
    imu_yaw_deg: Optional[float] = None
    imu_pitch_deg: Optional[float] = None
    imu_roll_deg: Optional[float] = None
    climate_activity: Optional[float] = None
    particulate_activity: Optional[float] = None
    em_activity: Optional[float] = None
    uv_activity: Optional[float] = None
    vibration_activity: Optional[float] = None
    proximity_activity: Optional[float] = None


class RuntimeContextV1(BaseModel):
    """Which LLM is actually generating this reply, for the situation brief.

    Juniper asked (2026-08-13) whether Orion has any sense of what model it's
    running on -- it did not. Investigated end to end: orion-llm-gateway's
    `model_used` field on chat responses was echoing the requested route
    label (e.g. "Active-GGUF-Model"), not the real served weights, and the
    value never reached any prompt Orion sees. This model is that fix's
    prompt-facing half -- see `orion/situational/context.py`'s
    `_build_runtime_context()` for the producer, and `llm_backend.py`'s
    `_served_model()` / `route_catalog.py`'s `_probe_model()` in
    orion-llm-gateway for where the honest value comes from (a live
    `/v1/models` probe against the route's backend, not the route-table
    label).

    `available=False` is the honest default: probe failure, a disabled flag,
    or an unreachable gateway all degrade to "unavailable", not a guess.
    Mirrors `LabContextV1`/`PerceptionContextV1`'s pattern in this file.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool = False
    route: str = "chat"
    model_id: Optional[str] = None
    served_by: Optional[str] = None
    backend: Optional[str] = None
    source: str = "none"


class SurfaceContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface: Literal["hub_desktop", "hub_mobile", "voice", "social_room", "operator_review", "scheduled", "unknown"] = "unknown"
    input_modality: Literal["typed", "spoken", "external_room", "scheduled", "unknown"] = "unknown"
    output_constraints: list[Literal["prefer_short", "hands_free", "high_interruptibility"]] = Field(default_factory=list)


class SituationAffordanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "temporal_resume",
        "outdoor_departure",
        "driving_or_travel",
        "kid_friendly_explanation",
        "family_audience",
        "privacy_sensitive_audience",
        "late_night_risk",
        "lab_thermal_risk",
        "lab_power_risk",
        "calendar_constraint",
        "weather_alert",
        "stale_context_revalidation",
        "fatigue_or_sleep_boundary",
    ]
    trigger_relevance: Literal["active", "only_if_user_mentions", "background", "suppressed"] = "background"
    suggestion: str
    confidence: Literal["low", "medium", "high"] = "medium"
    source_fields: list[str] = Field(default_factory=list)


class SituationPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    do_not_force_into_reply: bool = True
    use_only_when_relevant: bool = True
    avoid_exact_location_unless_needed: bool = True
    session_presence_not_memory_by_default: bool = True
    child_presence_requires_age_appropriate_style: bool = True
    do_not_infer_private_facts_from_presence: bool = True
    do_not_overpersonalize: bool = True


class PerceptionContextV1(BaseModel):
    """What Orion's camera saw, summarised, for the situation brief.

    **The exposed-field list is the privacy contract, and it is short on
    purpose.** Only a natural-language scene summary and its age cross into the
    prompt by default. Deliberately absent, and not to be added without
    proposal-mode sign-off: raw frames or frame paths, bounding boxes,
    per-object detections, embeddings, names of anyone other than the one
    enrolled subject, and any raw identity artifact (`vision_events.entities`,
    faces, re-ID vectors). The perception design doc lists identity/face/re-ID
    as a non-goal for THAT surface; this schema is where that promise is kept
    or broken.

    **Narrowed 2026-08-26, Juniper's direct ask**: two coarse,
    already-hedged identity SIGNALS now do cross this boundary --
    `presence_subject` (a name, only ever the one enrolled subject, only ever
    on a probable/possible match -- see `presence.py`'s own docstring) and
    `presence_identity_uncertain` (a bare boolean, gated by a cross-process
    cooldown so it can drive at most one "is that you?" per sit-down, never a
    repeating one). Neither carries a frame, a box, an embedding, or a raw
    model score -- both are the SAME kind of derived, hedged fact this
    schema already exposed for camera presence duration before identity
    entered the picture at all. Everything else in the excluded list above
    remains excluded.

    `available=False` is a real state, not an error state: it is how "I have not
    seen anything recently" is represented, and it must render as exactly that
    rather than as a stale observation presented as current.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["perception.context.v1"] = "perception.context.v1"
    available: bool = False
    # Natural-language only. The vision council's narrative, e.g. "Three chairs,
    # two tables, one door and one screen are visible in the scene."
    scene_summary: Optional[str] = None
    observed_at: Optional[datetime] = None
    observation_age_seconds: Optional[int] = None
    stream_id: Optional[str] = None
    # "live" | "stale" | "disabled" | "unavailable" | "error" -- why the caller
    # is or is not getting a percept, so a missing one is never ambiguous.
    source: str = "unavailable"
    # Camera-derived content about a private home. session_only is the only
    # value this should hold without an explicit, separately-approved change.
    privacy_mode: Literal["session_only", "persist_allowed"] = "session_only"

    # Embodied presence (orion-vision-window's `substrate_embodied_presence`,
    # see that service's app/presence.py) folded into `scene_summary` as a
    # sentence fragment -- "someone has been in view for..." (never a name,
    # by that fragment builder's own design) -- AND exposed here as small
    # structured fields for a non-prompt consumer (a debug surface, say)
    # that would rather read a number than parse prose.
    #
    # `presence_subject` is "unknown" or "none" UNLESS `identity_face` has
    # produced a fresh probable/possible match for the one enrolled subject,
    # in which case it is that subject's real name -- see `presence.py`'s
    # own docstring for the exact narrowing rule and its staleness/stickiness
    # guards. `presence_identity_uncertain` is True only when a real person
    # is currently believed present AND the most recent identity_face read
    # genuinely did not match (not "never ran") -- see the class docstring
    # above for the 2026-08-26 scope note and `identity_ask_cooldown.py` for
    # the anti-repetition gate that decides whether this ever reaches True in
    # a given turn.
    presence_state: Optional[str] = None            # "present" | "recent" | "absent"
    presence_since_sec: Optional[float] = None
    presence_subject: Optional[str] = None           # "unknown" | "none" | a real name
    presence_identity_uncertain: bool = False

    # The ASK DECISION, 2026-08-29 -- distinct from the observation above.
    #
    # `presence_identity_uncertain` is one observation ("a face was seen and
    # did not match"). This field is the decision that survived the cooldown
    # claim, and it covers a strictly larger set of situations, because the
    # observation could never describe the case Juniper actually reported:
    # "orion never bites when they can't recognize me (eg I close the camera
    # lid)". A closed lid emits no frames, so no face is detected, so
    # `identity_confidence` is None rather than "uncertain", so the boolean
    # above is False -- Orion stayed silent precisely when it could see
    # nothing at all. Confirmed live the same day: three presence rows, all
    # `identity_uncertain=false`, and the cortex-exec chat replica was
    # reading `cam0` (absent 70 minutes) while `carbon` -- the laptop webcam
    # -- showed a person present.
    #
    #   "unmatched_face"        a person is in view and identity_face did
    #                           not match the one enrolled subject
    #   "identity_unread"       a person IS in view, fresh, but no identity
    #                           reading exists for them (no face in the
    #                           sampled frame, or an unenrolled gallery)
    #   "no_visual_confirmation" no fresh read at all. Note this names what
    #                           Orion LACKS, not a physical camera state:
    #                           a closed lid, a stalled presence writer and a
    #                           down vision stack are indistinguishable here,
    #                           and an earlier version asserted the first
    #   None                    either Orion has a fresh confirmed read, or
    #                           the cooldown for that reason is already held
    #
    # Kept separate from the boolean rather than replacing it: that field is
    # registered in orion/schemas/registry.py and consumed as a structured
    # debug signal, and widening its meaning in place would silently change
    # what an existing reader thinks it is looking at. Both are set
    # consistently -- reason "unmatched_face" implies the boolean is True.
    # The prompt reads ONLY this field.
    presence_identity_ask: Optional[
        Literal["unmatched_face", "identity_unread", "no_visual_confirmation"]
    ] = None


class AffectContextV1(BaseModel):
    """Juniper's most recent facial+vocal affect read, for the situation brief.

    Source (since 2026-08-26): a VL read of the clip's own frames via
    `orion-llm-gateway`, relayed through
    `orion-juniper-affective-state` (`JuniperMultimodalAffectV1`, published on
    `orion:affectgpt:assessment`) and mirrored into a single Redis key by
    `orion/situational/juniper_affect_state.py` -- see that module's
    docstring for the write side. This schema is the read side's privacy
    contract, same role `PerceptionContextV1`'s docstring plays for camera
    content.

    **`summary` is a short rendered line built from the model's STRUCTURED
    read (`AffectReadV1`), never the verbatim spoken transcript, and -- since
    2026-08-26 -- no longer the model's raw prose either.** Passing raw prose
    is what put "it is not possible to infer the character's emotional state
    from the subtitle content" into Juniper's chat prompt for turn ddddfe40. Juniper's actual
    words (`JuniperMultimodalAffectV1.transcript`) are deliberately NOT
    forwarded into a chat prompt -- Orion gets the model's inferred
    affect description, not a transcript of private speech. Do not widen
    this without proposal-mode sign-off (CLAUDE.md's cognition-change gate).

    `available=False` is a real state ("no recent capture" / "capture
    failed" / "too old to trust"), not an error swallowed into silence --
    same honesty contract as `PerceptionContextV1.available`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["affect.context.v1"] = "affect.context.v1"
    available: bool = False
    summary: Optional[str] = None
    observed_at: Optional[datetime] = None
    observation_age_seconds: Optional[int] = None
    # "manual" (Hub's "Check now" button) | "ambient" (Hub's ambient toggle,
    # ~5min cadence) -- lets the prompt distinguish "you just asked to be
    # seen" from "this is from the background loop".
    trigger: Optional[str] = None
    # Mirrors AffectGptAssessResultPayload.subtitle_source -- "transcribed"
    # means Whisper actually heard something; "none" means silence/no audio
    # signal reached the model at all, which materially changes how much to
    # trust the read.
    subtitle_source: Optional[str] = None
    # The producing model's own confidence, 0.0-1.0. Only ever populated by
    # backend="vision" (2026-08-26); the retired affectgpt backend reported
    # none. Present so the prompt line can hedge PROPORTIONALLY rather than
    # presenting every surviving read with identical certainty -- the
    # write-side gate has already dropped everything below
    # AFFECT_MIRROR_MIN_CONFIDENCE, so what arrives here is above the bar,
    # which is not the same as being sure.
    confidence: Optional[float] = None
    # Which inference backend produced the read: "vision" | "affectgpt" |
    # None (a payload written before the field existed). Carried so a debug
    # surface can attribute a read without joining back to the event log.
    backend: Optional[str] = None
    # "live" | "stale" | "disabled" | "unavailable" | "error" -- same
    # vocabulary as PerceptionContextV1.source, so a missing read is never
    # ambiguous about why.
    source: str = "unavailable"
    # Juniper's own inferred emotional state. session_only is the only value
    # this should hold without an explicit, separately-approved change --
    # same posture as PerceptionContextV1.privacy_mode.
    privacy_mode: Literal["session_only", "persist_allowed"] = "session_only"


class CuriosityPriorSummaryV1(BaseModel):
    """One `:Prior` node from Orion's own `orion_worldview` graph, hedged for
    the prompt. See `CuriosityPriorContextV1`'s docstring for the exposed-
    field contract this deliberately stays inside."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    confidence: Optional[float] = None
    status: str = "open"
    times_tested: int = 0


class CuriosityPriorContextV1(BaseModel):
    """Orion's own open world-priors, from Orion's `orion_worldview` FalkorDB
    graph (`:Prior` nodes), for the situation brief.

    Read via `orion.curiosity.worldview.WorldviewReader.read_snapshot()` --
    the exact same producer `services/orion-hub/scripts/
    curiosity_investigation.py`'s self-study loop already uses to decide
    what to test next. This is a SEPARATE consumer of that same read: the
    loop reads it to pick a prior to investigate; this reads it so a live
    chat turn can carry a short, already-hedged flavor of what Orion
    currently believes and how sure it is -- the same "grounding, not
    gospel" role `AffectContextV1`/`PerceptionContextV1` play for camera
    and mood.

    Deliberately narrow: `summaries` carries only `claim`/`confidence`/
    `status`/`times_tested` -- never `formed_from` (may reference private
    material) or `prior_id` (an internal graph handle with no prompt use).
    Capped small (a handful of priors, ranked by confidence -- see
    `orion/situational/context.py`'s `_fetch_curiosity_context`) to respect
    the prompt budget; this is color, not a dump of Orion's whole
    worldview.

    Not the endogenous-curiosity/Postgres candidate system
    (`orion/substrate/endogenous_curiosity.py`) or `curiosity_hint.py` --
    those are a different producer with a different consumer. This is
    Orion's own graph-backed world-priors, read read-only via
    `GRAPH.RO_QUERY`.

    `available=False` is a real state (disabled / graph not configured /
    empty live pool / read failure) -- same honesty contract as
    `PerceptionContextV1.available`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["curiosity.prior_context.v1"] = "curiosity.prior_context.v1"
    available: bool = False
    summaries: list[CuriosityPriorSummaryV1] = Field(default_factory=list)
    live_total: Optional[int] = None
    # "orion_worldview" (live) | "disabled" | "unconfigured" | "unavailable" | "error"
    source: str = "unavailable"


class ReverieSnippetV1(BaseModel):
    """One short reverie/dream interpretation. See `ReverieContextV1`'s
    docstring for the exposed-field contract this deliberately stays
    inside."""

    model_config = ConfigDict(extra="forbid")

    text: str
    observed_at: Optional[datetime] = None
    salience: Optional[float] = None


class ReverieContextV1(BaseModel):
    """Orion's most recent dream/reverie interpretations, for the situation
    brief.

    Source: a plain SQL read of `substrate_reverie_thought` (Postgres) via
    `orion/situational/reverie_reader.py` -- the same table Hub's reverie
    cockpit (`services/orion-hub/scripts/reverie_routes.py`'s
    `/api/reverie/text/recent` route) already renders for Juniper. This is a
    second, narrower reader of that same table rather than an import of the
    route handler's own fetch function -- `orion/` is shared code services
    import FROM, so a route script in `services/orion-hub/` must not be
    imported the other way round.

    Deliberately NOT wired to the `orion:reverie:thought`/`orion:reverie:
    chain` bus channels: those have zero real subscribers today, and a plain
    SQL read of a table an established UI already trusts is a much thinner
    seam than standing up a new bus consumer for one small feature.

    Only the model's own short interpretation text, its age, and its
    salience score cross into the prompt -- never raw generated image bytes/
    paths, chain linkage, or diffusion-prompt internals
    `reverie_visual_chain` carries.

    `available=False` is a real state (disabled / no DSN configured / no
    rows yet / read failure), not an error swallowed into silence -- same
    honesty contract as `PerceptionContextV1.available`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["reverie.context.v1"] = "reverie.context.v1"
    available: bool = False
    snippets: list[ReverieSnippetV1] = Field(default_factory=list)
    # "reverie_sql" (live) | "disabled" | "unavailable" | "error"
    source: str = "unavailable"


class SituationDiagnosticsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_status: dict[str, str] = Field(default_factory=dict)
    provider_errors: dict[str, str] = Field(default_factory=dict)
    relevance_reasons: list[str] = Field(default_factory=list)
    generated_with_partial_context: bool = False


class SituationBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["situation.brief.v1"] = "situation.brief.v1"
    generated_at: datetime
    ttl_seconds: int = 300
    source_summary: dict[str, str] = Field(default_factory=dict)
    requestor: RequestorContextV1 = Field(default_factory=RequestorContextV1)
    presence: PresenceContextV1 = Field(default_factory=PresenceContextV1)
    time: TimeContextV1
    conversation_phase: ConversationPhaseContextV1
    place: PlaceContextV1
    environment: EnvironmentContextV1 = Field(default_factory=EnvironmentContextV1)
    agenda: AgendaContextV1 = Field(default_factory=AgendaContextV1)
    lab: LabContextV1 = Field(default_factory=LabContextV1)
    # Additive (2026-08-31): defaults to available=False. Rendered like
    # curiosity/reverie (omitted from the prompt entirely when unavailable,
    # not an always-on placeholder line) rather than like weather/lab/
    # perception -- see `_build_prompt_fragment`'s own comment for why. ON
    # by default in orion-hub (the only process with the `/run/orion-
    # sensors` bind mount) -- carries no private-home content, same
    # reasoning as curiosity/reverie above.
    cabinet: CabinetContextV1 = Field(default_factory=CabinetContextV1)
    # Additive: defaults to available=False, so an unpatched producer or a
    # disabled flag yields "haven't seen anything recently" rather than a
    # missing field.
    perception: PerceptionContextV1 = Field(default_factory=PerceptionContextV1)
    # Additive (2026-08-25): defaults to available=False, so an unpatched
    # producer/disabled flag/no-recent-capture yields "no recent affect
    # read" rather than a missing field or a stale guess.
    affect: AffectContextV1 = Field(default_factory=AffectContextV1)
    # Additive (2026-08-14): defaults to available=False, so an unpatched
    # producer/disabled flag/probe failure yields "do not infer" rather than
    # a missing field or a stale guess.
    runtime: RuntimeContextV1 = Field(default_factory=RuntimeContextV1)
    # Additive (2026-08-30): defaults to available=False, so an unpatched
    # producer/disabled flag/unconfigured graph/empty pool yields "do not
    # infer" rather than a missing field. ON by default (Juniper's explicit
    # call) -- unlike perception, this carries no private-home content.
    curiosity: CuriosityPriorContextV1 = Field(default_factory=CuriosityPriorContextV1)
    # Additive (2026-08-30): same defaulting contract as curiosity above. ON
    # by default (Juniper's explicit call).
    reverie: ReverieContextV1 = Field(default_factory=ReverieContextV1)
    surface: SurfaceContextV1 = Field(default_factory=SurfaceContextV1)
    affordances: list[SituationAffordanceV1] = Field(default_factory=list)
    policy: SituationPolicyV1 = Field(default_factory=SituationPolicyV1)
    diagnostics: SituationDiagnosticsV1 = Field(default_factory=SituationDiagnosticsV1)


class SituationPromptFragmentV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["situation.prompt_fragment.v1"] = "situation.prompt_fragment.v1"
    generated_at: datetime
    summary_lines: list[str] = Field(default_factory=list)
    relevance_notes: list[str] = Field(default_factory=list)
    caution_lines: list[str] = Field(default_factory=list)
    should_mention: bool = False
    mention_policy: Literal["only_if_relevant", "safe_to_mention", "do_not_mention"] = "only_if_relevant"
    compact_text: str = ""
    source_brief_id: Optional[str] = None
    max_chars_applied: int = 1200
