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
