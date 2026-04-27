from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from orion.schemas.situation import (
    AgendaContextV1,
    ConversationPhaseContextV1,
    EnvironmentContextV1,
    LabContextV1,
    PlaceContextV1,
    PresenceContextV1,
    RequestorContextV1,
    SituationAffordanceV1,
    SituationBriefV1,
    SituationDiagnosticsV1,
    SituationPromptFragmentV1,
    SurfaceContextV1,
    TimeContextV1,
    WeatherCurrentV1,
    WeatherForecastWindowV1,
    WeatherPracticalFlagsV1,
)


_LOCK = threading.Lock()
_SITUATION_CACHE: dict[str, tuple[datetime, SituationBriefV1, SituationPromptFragmentV1]] = {}
_WEATHER_CACHE: dict[str, tuple[datetime, EnvironmentContextV1]] = {}
_SESSION_LAST_USER_TURN: dict[str, datetime] = {}
_SESSION_LAST_ORION_TURN: dict[str, datetime] = {}


@dataclass
class SituationSettings:
    enabled: bool
    ttl_seconds: int
    prompt_max_chars: int
    timezone: str
    location_label: str
    locality: str | None
    region: str | None
    country: str | None
    location_precision: str
    weather_enabled: bool
    weather_provider: str
    weather_lat: float | None
    weather_lon: float | None
    weather_ttl_seconds: int
    umbrella_prob_threshold: int
    jacket_temp_f_threshold: int
    high_wind_mph_threshold: int
    hot_car_temp_f_threshold: int
    agenda_enabled: bool
    lab_enabled: bool
    lab_provider: str
    default_requestor: str
    presence_persist_allowed: bool


def settings_from_runtime(settings: Any) -> SituationSettings:
    return SituationSettings(
        enabled=bool(getattr(settings, "orion_situation_enabled", True)),
        ttl_seconds=int(getattr(settings, "orion_situation_ttl_seconds", 300)),
        prompt_max_chars=int(getattr(settings, "orion_situation_prompt_max_chars", 1200)),
        timezone=str(getattr(settings, "orion_situation_timezone", "America/Denver")),
        location_label=str(getattr(settings, "orion_situation_location_label", "Unknown")),
        locality=getattr(settings, "orion_situation_locality", None),
        region=getattr(settings, "orion_situation_region", None),
        country=getattr(settings, "orion_situation_country", None),
        location_precision=str(getattr(settings, "orion_situation_location_precision", "city")),
        weather_enabled=bool(getattr(settings, "orion_situation_weather_enabled", True)),
        weather_provider=str(getattr(settings, "orion_situation_weather_provider", "stub")),
        weather_lat=getattr(settings, "orion_situation_weather_lat", None),
        weather_lon=getattr(settings, "orion_situation_weather_lon", None),
        weather_ttl_seconds=int(getattr(settings, "orion_situation_weather_ttl_seconds", 600)),
        umbrella_prob_threshold=int(getattr(settings, "orion_situation_umbrella_precip_prob_threshold", 40)),
        jacket_temp_f_threshold=int(getattr(settings, "orion_situation_jacket_temp_f_threshold", 55)),
        high_wind_mph_threshold=int(getattr(settings, "orion_situation_high_wind_mph_threshold", 25)),
        hot_car_temp_f_threshold=int(getattr(settings, "orion_situation_hot_car_temp_f_threshold", 80)),
        agenda_enabled=bool(getattr(settings, "orion_situation_agenda_enabled", False)),
        lab_enabled=bool(getattr(settings, "orion_situation_lab_context_enabled", True)),
        lab_provider=str(getattr(settings, "orion_situation_lab_provider", "stub")),
        default_requestor=str(getattr(settings, "orion_presence_default_requestor", "Juniper")),
        presence_persist_allowed=bool(getattr(settings, "orion_presence_persist_allowed", False)),
    )


def build_situation_for_ctx(ctx: dict[str, Any], runtime_settings: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = settings_from_runtime(runtime_settings)
    if not cfg.enabled:
        return {}, {}
    cache_key = str(ctx.get("session_id") or "global")
    with _LOCK:
        cached = _SITUATION_CACHE.get(cache_key)
        if cached and (datetime.now(UTC) - cached[0]).total_seconds() < cfg.ttl_seconds:
            return cached[1].model_dump(mode="json"), cached[2].model_dump(mode="json")

    now = datetime.now(UTC)
    diagnostics = SituationDiagnosticsV1()
    presence = _presence_from_ctx(ctx, cfg, now)
    time_ctx = _build_time_context(cfg, diagnostics)
    phase_ctx = _build_conversation_phase(ctx, time_ctx, now)
    place_ctx = _build_place_context(cfg)
    env_ctx = _build_environment_context(cfg, diagnostics)
    agenda_ctx = AgendaContextV1(available=False, source="stub")
    lab_ctx = _build_lab_context(cfg)
    surface_ctx = _build_surface_context(ctx)
    affordances = _build_affordances(ctx, presence, phase_ctx, env_ctx, lab_ctx, surface_ctx, time_ctx)
    diagnostics.relevance_reasons = [a.kind for a in affordances if a.trigger_relevance == "active"]

    brief = SituationBriefV1(
        generated_at=now,
        ttl_seconds=cfg.ttl_seconds,
        source_summary={
            "weather": env_ctx.source,
            "presence": presence.source,
            "phase": phase_ctx.phase_change,
            "surface": surface_ctx.surface,
        },
        requestor=presence.requestor,
        presence=presence,
        time=time_ctx,
        conversation_phase=phase_ctx,
        place=place_ctx,
        environment=env_ctx,
        agenda=agenda_ctx,
        lab=lab_ctx,
        surface=surface_ctx,
        affordances=affordances,
        diagnostics=diagnostics,
    )
    fragment = _build_prompt_fragment(brief, cfg.prompt_max_chars)
    with _LOCK:
        _SITUATION_CACHE[cache_key] = (now, brief, fragment)
    return brief.model_dump(mode="json"), fragment.model_dump(mode="json")


def _presence_from_ctx(ctx: dict[str, Any], cfg: SituationSettings, now: datetime) -> PresenceContextV1:
    raw = ctx.get("presence_context") if isinstance(ctx.get("presence_context"), dict) else {}
    requestor = RequestorContextV1(
        display_name=str((raw.get("requestor") or {}).get("display_name") or cfg.default_requestor),
        relationship_to_orion=str((raw.get("requestor") or {}).get("relationship_to_orion") or "primary_operator"),
        source=str((raw.get("requestor") or {}).get("source") or "hub_manual"),
        confidence=str((raw.get("requestor") or {}).get("confidence") or "medium"),
    )
    companions = raw.get("companions") if isinstance(raw.get("companions"), list) else []
    normalized_companions = []
    for item in companions[:8]:
        if not isinstance(item, dict):
            continue
        if not item.get("display_name"):
            continue
        normalized_companions.append(item)
    audience_mode = str(raw.get("audience_mode") or ("solo" if not normalized_companions else "mixed_group"))
    return PresenceContextV1(
        requestor=requestor,
        companions=normalized_companions,
        audience_mode=audience_mode,  # type: ignore[arg-type]
        submitted_at=now,
        expires_at=now + timedelta(hours=4),
        source=str(raw.get("source") or "default"),
        persist_to_memory=bool(raw.get("persist_to_memory", False) and cfg.presence_persist_allowed),
        privacy_mode=str(raw.get("privacy_mode") or "session_only"),  # type: ignore[arg-type]
        notes=str(raw.get("notes")) if raw.get("notes") else None,
    )


def _build_time_context(cfg: SituationSettings, diagnostics: SituationDiagnosticsV1) -> TimeContextV1:
    tz = ZoneInfo(cfg.timezone)
    now_local = datetime.now(tz)
    hour = now_local.hour
    minute = now_local.minute
    tod = _time_of_day_label(hour)
    day_phase = _day_phase_label(hour, minute)
    sunrise = None
    sunset = None
    sun_phase = "unknown"
    season = _season_label(now_local.month)
    return TimeContextV1(
        timezone=cfg.timezone,
        local_datetime=now_local.isoformat(),
        local_date=now_local.strftime("%Y-%m-%d"),
        local_time=now_local.strftime("%H:%M"),
        weekday=now_local.strftime("%A"),
        is_weekend=now_local.weekday() >= 5,
        season_local=season,
        time_of_day_label=tod,
        day_phase=day_phase,
        sun_phase=sun_phase,
        sunrise_local=sunrise,
        sunset_local=sunset,
    )


def _time_of_day_label(hour: int) -> str:
    if hour < 5:
        return "pre_dawn"
    if hour < 8:
        return "early_morning"
    if hour < 10:
        return "mid_morning"
    if hour < 12:
        return "late_morning"
    if hour < 14:
        return "midday"
    if hour < 16:
        return "early_afternoon"
    if hour < 18:
        return "late_afternoon"
    if hour < 21:
        return "evening"
    if hour < 23:
        return "late_evening"
    return "night"


def _day_phase_label(hour: int, minute: int) -> str:
    hm = hour * 60 + minute
    if hm < 300:
        return "pre_dawn"
    if hm < 420:
        return "dawn"
    if hm < 720:
        return "morning"
    if hm < 840:
        return "midday"
    if hm < 1080:
        return "afternoon"
    if hm < 1200:
        return "dusk"
    return "night"


def _season_label(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _build_conversation_phase(ctx: dict[str, Any], time_ctx: TimeContextV1, now_utc: datetime) -> ConversationPhaseContextV1:
    session_id = str(ctx.get("session_id") or "global")
    with _LOCK:
        last_user = _SESSION_LAST_USER_TURN.get(session_id)
        last_orion = _SESSION_LAST_ORION_TURN.get(session_id)
    delta_user = int((now_utc - last_user).total_seconds()) if last_user else None
    phase = "unknown"
    continuity = "continue_directly"
    risk = "none"
    adjustments: list[str] = []
    crossed_day = False
    if last_user:
        crossed_day = last_user.astimezone(ZoneInfo(time_ctx.timezone)).date() != datetime.now(
            ZoneInfo(time_ctx.timezone)
        ).date()
        if delta_user is not None and delta_user < 120:
            phase = "same_breath"
        elif delta_user < 20 * 60:
            phase = "short_pause"
        elif delta_user < 3 * 3600:
            phase = "resumed_thread"
            continuity = "lightly_resume"
            risk = "low"
        elif delta_user < 12 * 3600:
            phase = "long_gap"
            continuity = "reorient"
            risk = "medium"
            adjustments.append("Reorient before acting on stale operational context.")
        elif delta_user > 48 * 3600:
            phase = "stale_thread"
            continuity = "revalidate_context"
            risk = "high"
            adjustments.append("Revalidate volatile assumptions and preflight checks.")
        if crossed_day and phase not in {"stale_thread", "unknown"}:
            phase = "next_day"
            continuity = "reorient"
            risk = "medium"
            adjustments.append("Crossed day boundary; lightly re-anchor timeline.")
    out = ConversationPhaseContextV1(
        last_user_turn_at=last_user,
        last_orion_turn_at=last_orion,
        time_since_last_user_turn_seconds=delta_user,
        time_since_last_orion_turn_seconds=int((now_utc - last_orion).total_seconds()) if last_orion else None,
        crossed_day_boundary=crossed_day,
        phase_change=phase,  # type: ignore[arg-type]
        continuity_mode=continuity,  # type: ignore[arg-type]
        topic_staleness_risk=risk,  # type: ignore[arg-type]
        response_adjustments=adjustments,
    )
    with _LOCK:
        _SESSION_LAST_USER_TURN[session_id] = now_utc
    return out


def mark_orion_turn(session_id: str | None) -> None:
    with _LOCK:
        _SESSION_LAST_ORION_TURN[str(session_id or "global")] = datetime.now(UTC)


def _build_place_context(cfg: SituationSettings) -> PlaceContextV1:
    return PlaceContextV1(
        coarse_location=cfg.location_label,
        locality=cfg.locality,
        region=cfg.region,
        country=cfg.country,
        timezone=cfg.timezone,
        precision=cfg.location_precision,  # type: ignore[arg-type]
        source="configured_home" if cfg.location_label != "Unknown" else "unknown",
        confidence="medium" if cfg.location_label != "Unknown" else "low",
    )


def _build_environment_context(cfg: SituationSettings, diagnostics: SituationDiagnosticsV1) -> EnvironmentContextV1:
    if not cfg.weather_enabled:
        diagnostics.provider_status["weather"] = "disabled"
        return EnvironmentContextV1(available=False, source="disabled")
    cache_key = f"{cfg.weather_provider}:{cfg.weather_lat}:{cfg.weather_lon}"
    with _LOCK:
        cached = _WEATHER_CACHE.get(cache_key)
        if cached and (datetime.now(UTC) - cached[0]).total_seconds() < cfg.weather_ttl_seconds:
            return cached[1]
    try:
        env = _fetch_weather(cfg)
        with _LOCK:
            _WEATHER_CACHE[cache_key] = (datetime.now(UTC), env)
        diagnostics.provider_status["weather"] = "ok"
        return env
    except Exception as exc:
        diagnostics.provider_status["weather"] = "error"
        diagnostics.provider_errors["weather"] = str(exc)
        return EnvironmentContextV1(available=False, source="error")


def _fetch_weather(cfg: SituationSettings) -> EnvironmentContextV1:
    provider = cfg.weather_provider.lower().strip()
    if provider in {"none", "stub"}:
        return EnvironmentContextV1(available=False, source=provider)
    if provider != "openmeteo" or cfg.weather_lat is None or cfg.weather_lon is None:
        return EnvironmentContextV1(available=False, source="misconfigured")
    params = urlencode(
        {
            "latitude": cfg.weather_lat,
            "longitude": cfg.weather_lon,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m",
            "hourly": "precipitation_probability,temperature_2m,wind_speed_10m",
            "forecast_days": 2,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
        }
    )
    with urlopen(f"https://api.open-meteo.com/v1/forecast?{params}", timeout=4) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    current = payload.get("current") or {}
    hourly = payload.get("hourly") or {}
    probs = hourly.get("precipitation_probability") or []
    temps = hourly.get("temperature_2m") or []
    winds = hourly.get("wind_speed_10m") or []
    flag = WeatherPracticalFlagsV1(
        take_umbrella=_window_max(probs, 6) >= cfg.umbrella_prob_threshold,
        take_jacket=(current.get("temperature_2m") or 999) <= cfg.jacket_temp_f_threshold,
        high_wind=(current.get("wind_speed_10m") or 0) >= cfg.high_wind_mph_threshold,
        icy_roads=(_window_min(temps, 6) <= 32 and _window_max(probs, 6) >= 20),
        hot_car_risk=_window_max(temps, 6) >= cfg.hot_car_temp_f_threshold,
        severe_weather=False,
    )
    out = EnvironmentContextV1(
        available=True,
        source="openmeteo",
        confidence="medium",
        source_age_seconds=0,
        current_weather=WeatherCurrentV1(
            temperature_f=_to_float(current.get("temperature_2m")),
            feels_like_f=_to_float(current.get("apparent_temperature")),
            condition=f"code:{current.get('weather_code', 'unknown')}",
            wind_mph=_to_float(current.get("wind_speed_10m")),
            wind_gust_mph=_to_float(current.get("wind_gusts_10m")),
        ),
        forecast_next_2h=WeatherForecastWindowV1(
            window_label="next_2h",
            summary="Near-term conditions",
            precipitation_probability_pct=_window_max(probs, 2),
            wind_max_mph=_to_float(_window_max(winds, 2)),
            temperature_low_f=_to_float(_window_min(temps, 2)),
            temperature_high_f=_to_float(_window_max(temps, 2)),
        ),
        forecast_next_6h=WeatherForecastWindowV1(
            window_label="next_6h",
            summary="Short-range weather window",
            precipitation_probability_pct=_window_max(probs, 6),
            wind_max_mph=_to_float(_window_max(winds, 6)),
            temperature_low_f=_to_float(_window_min(temps, 6)),
            temperature_high_f=_to_float(_window_max(temps, 6)),
        ),
        forecast_next_24h=WeatherForecastWindowV1(
            window_label="next_24h",
            summary="Day weather window",
            precipitation_probability_pct=_window_max(probs, 24),
            wind_max_mph=_to_float(_window_max(winds, 24)),
            temperature_low_f=_to_float(_window_min(temps, 24)),
            temperature_high_f=_to_float(_window_max(temps, 24)),
        ),
        practical_flags=flag,
    )
    return out


def _window_max(values: list[Any], count: int) -> int:
    nums = [_to_float(v) for v in values[:count] if _to_float(v) is not None]
    return int(max(nums)) if nums else 0


def _window_min(values: list[Any], count: int) -> int:
    nums = [_to_float(v) for v in values[:count] if _to_float(v) is not None]
    return int(min(nums)) if nums else 0


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _build_lab_context(cfg: SituationSettings) -> LabContextV1:
    if not cfg.lab_enabled:
        return LabContextV1(available=False, source="disabled")
    return LabContextV1(available=False, source=cfg.lab_provider, thermal_risk="unknown", power_risk="unknown")


def _build_surface_context(ctx: dict[str, Any]) -> SurfaceContextV1:
    md = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    surface = str((md.get("surface_context") or {}).get("surface") or "hub_desktop")
    input_modality = str((md.get("surface_context") or {}).get("input_modality") or "typed")
    return SurfaceContextV1(surface=surface, input_modality=input_modality)  # type: ignore[arg-type]


def _build_affordances(
    ctx: dict[str, Any],
    presence: PresenceContextV1,
    phase: ConversationPhaseContextV1,
    env: EnvironmentContextV1,
    lab: LabContextV1,
    surface: SurfaceContextV1,
    time_ctx: TimeContextV1,
) -> list[SituationAffordanceV1]:
    text = str(ctx.get("raw_user_text") or ctx.get("user_message") or "").lower()
    out: list[SituationAffordanceV1] = []
    if any(k in text for k in ("heading out", "leaving", "going outside", "driving", "trip", "airport", "commute")):
        out.append(
            SituationAffordanceV1(
                kind="outdoor_departure",
                trigger_relevance="active",
                suggestion="User appears to be departing; weather practicality can be briefly used if asked.",
                confidence="high",
                source_fields=["environment.forecast_next_6h", "environment.practical_flags"],
            )
        )
    if phase.phase_change in {"long_gap", "next_day", "stale_thread"}:
        out.append(
            SituationAffordanceV1(
                kind="temporal_resume",
                trigger_relevance="active",
                suggestion="Thread resumed after temporal phase change; reorient or revalidate before volatile actions.",
                confidence="high",
                source_fields=["conversation_phase.phase_change"],
            )
        )
    if presence.audience_mode in {"kid_present", "family"} or any(c.age_band == "child" for c in presence.companions):
        out.append(
            SituationAffordanceV1(
                kind="kid_friendly_explanation",
                trigger_relevance="active",
                suggestion="Child listener/asker present; prefer clear age-appropriate explanation when relevant.",
                confidence="medium",
                source_fields=["presence.audience_mode", "presence.companions"],
            )
        )
    if env.available and env.practical_flags.severe_weather:
        out.append(
            SituationAffordanceV1(
                kind="weather_alert",
                trigger_relevance="active",
                suggestion="Severe weather active; mention only if it changes user safety/decision.",
                confidence="high",
                source_fields=["environment.weather_alerts", "environment.practical_flags"],
            )
        )
    if time_ctx.time_of_day_label in {"late_evening", "night"} and any(
        k in text for k in ("migration", "deploy", "delete", "prune", "partition", "sudo")
    ):
        out.append(
            SituationAffordanceV1(
                kind="late_night_risk",
                trigger_relevance="active",
                suggestion="Late-night risky operation; suggest preflight and rollback plan briefly.",
                confidence="medium",
                source_fields=["time.time_of_day_label", "user_message"],
            )
        )
    if not out:
        out.append(
            SituationAffordanceV1(
                kind="fatigue_or_sleep_boundary",
                trigger_relevance="background",
                suggestion="Use situation context only when materially relevant.",
                confidence="low",
                source_fields=["policy"],
            )
        )
    return out


def _build_prompt_fragment(brief: SituationBriefV1, max_chars: int) -> SituationPromptFragmentV1:
    lines = [
        f"Local context: {brief.time.time_of_day_label.replace('_', ' ')} {brief.time.weekday}, {brief.time.timezone}.",
        f"Conversation phase: {brief.conversation_phase.phase_change}; continuity={brief.conversation_phase.continuity_mode}.",
        f"Presence: requestor={brief.requestor.display_name}, audience_mode={brief.presence.audience_mode}.",
    ]
    if brief.environment.available:
        rain = brief.environment.forecast_next_6h.precipitation_probability_pct
        lines.append(f"Weather next 6h: precip_prob={rain}%, summary={brief.environment.forecast_next_6h.summary}.")
    else:
        lines.append("Weather: unavailable or low-confidence; do not infer.")
    if brief.lab.available:
        lines.append(f"Lab risk: thermal={brief.lab.thermal_risk}, power={brief.lab.power_risk}.")
    else:
        lines.append("Lab: unavailable/stub; do not infer.")
    relevance = [f"{a.kind}: {a.suggestion}" for a in brief.affordances if a.trigger_relevance == "active"]
    cautions = [
        "Situation context is grounding, not a requirement to mention.",
        "Use only when relevant; avoid contrived time/weather/location commentary.",
    ]
    compact = "Situation:\n- " + "\n- ".join(lines + relevance + cautions)
    if len(compact) > max_chars:
        compact = compact[: max_chars - 1] + "…"
    return SituationPromptFragmentV1(
        generated_at=brief.generated_at,
        summary_lines=lines,
        relevance_notes=relevance,
        caution_lines=cautions,
        should_mention=bool(relevance),
        mention_policy="only_if_relevant",
        compact_text=compact,
        max_chars_applied=max_chars,
    )
