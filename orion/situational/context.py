from __future__ import annotations

import asyncio
import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from .identity_ask_cooldown import try_claim_identity_ask
from .juniper_affect_state import read_latest_juniper_affect
from .perception_reader import (
    coarse_duration,
    fetch_latest_percept,
    fetch_presence,
    percept_age_seconds,
    presence_fragment,
)
from .session_turn_phase import read_session_turn_state, write_session_turn_state
from orion.schemas.situation import (
    AffectContextV1,
    AgendaContextV1,
    ConversationPhaseContextV1,
    EnvironmentContextV1,
    LabContextV1,
    PerceptionContextV1,
    PlaceContextV1,
    PresenceContextV1,
    RequestorContextV1,
    RuntimeContextV1,
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
_RUNTIME_CACHE: dict[str, tuple[datetime, RuntimeContextV1]] = {}
# NOTE: session turn-timestamp state (_SESSION_LAST_USER_TURN /
# _SESSION_LAST_ORION_TURN, formerly in-process dicts here) now lives in
# Redis via session_turn_phase.py -- see that module's docstring for why.
# _SITUATION_CACHE/_WEATHER_CACHE/_RUNTIME_CACHE above are unaffected: they
# are legitimate per-process TTL caches for external calls with no
# cross-process consistency requirement, not the bug this fixes.


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
    perception_enabled: bool
    perception_max_age_seconds: int
    perception_stream_id: str
    identity_ask_cooldown_seconds: int
    affect_enabled: bool
    affect_max_age_seconds: int
    runtime_enabled: bool
    runtime_route: str
    runtime_ttl_seconds: int
    runtime_probe_timeout_sec: float
    llm_gateway_base_url: str
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
        # Default OFF: this puts camera-derived content about a private home
        # into the prompt, so it is opt-in rather than opt-out.
        perception_enabled=bool(
            getattr(settings, "orion_situation_perception_enabled", False)
        ),
        # 900s. Live vision_events arrive roughly every 5 min on a static scene
        # (measured 2026-08-13), so this tolerates a few missed windows without
        # letting an hour-old percept read as current.
        perception_max_age_seconds=int(
            getattr(settings, "orion_situation_perception_max_age_seconds", 900)
        ),
        perception_stream_id=str(
            getattr(settings, "orion_situation_perception_stream_id", "cam0")
        ),
        # 1200s (20min): see identity_ask_cooldown.py's module docstring for
        # the reasoning -- long enough that "ask once per sit-down" is the
        # felt experience, short enough that a fixed lighting/angle issue
        # doesn't leave someone silently mis-recognized all day.
        identity_ask_cooldown_seconds=int(
            getattr(settings, "orion_situation_identity_ask_cooldown_seconds", 1200)
        ),
        # Default ON, unlike perception: the capture that produces this is
        # already an explicit Juniper action (Hub's "Check now"/ambient
        # toggle), so folding the result into the prompt is not new
        # surveillance, just surfacing what was already deliberately
        # captured. See juniper_affect_state.py + AffectContextV1's
        # docstrings for the privacy contract (excerpt only, never the
        # verbatim transcript).
        affect_enabled=bool(getattr(settings, "orion_situation_affect_enabled", True)),
        # 300s: matches Hub's ambient-capture cadence (~5min). Tighter than
        # perception's 900s on purpose -- a stale mood read is more likely
        # to mislead a reply than a stale room description is.
        affect_max_age_seconds=int(
            getattr(settings, "orion_situation_affect_max_age_seconds", 300)
        ),
        lab_provider=str(getattr(settings, "orion_situation_lab_provider", "stub")),
        runtime_enabled=bool(getattr(settings, "orion_situation_runtime_enabled", True)),
        runtime_route=str(getattr(settings, "orion_situation_runtime_route", "chat")),
        runtime_ttl_seconds=int(getattr(settings, "orion_situation_runtime_ttl_seconds", 120)),
        runtime_probe_timeout_sec=float(
            getattr(settings, "orion_situation_runtime_probe_timeout_sec", 2.0)
        ),
        llm_gateway_base_url=str(
            getattr(settings, "cortex_exec_llm_gateway_url", "http://llm-gateway:8210")
        ),
        default_requestor=str(getattr(settings, "orion_presence_default_requestor", "Juniper")),
        presence_persist_allowed=bool(getattr(settings, "orion_presence_persist_allowed", False)),
    )


def hub_settings_to_runtime_namespace(cfg: Any) -> SimpleNamespace:
    """Adapt orion-hub's `Settings` object into the lowercase-attribute shape
    `settings_from_runtime` (above) already expects.

    orion-hub's `Settings` class (services/orion-hub/app/settings.py)
    declares its own `ORION_SITUATION_*`/`ORION_PRESENCE_*` fields
    UPPERCASE, matching that file's existing local convention (and its own
    live consumers -- services/orion-hub/scripts/api_routes.py's
    `/api/situation/*` routes already read them that way; do not rename).
    `settings_from_runtime` reads lowercase attribute names via `getattr`,
    matching cortex-exec's `Settings` convention instead -- passing an
    UPPERCASE settings object straight in would silently miss every field
    (getattr is case-sensitive) and fall back to hardcoded literal defaults
    with no visible error. This bridges the two conventions explicitly
    rather than making `settings_from_runtime` guess casings.

    Fields orion-hub does not yet configure (location label/locality/
    region/country, lab, perception) are turned off here on purpose, not
    left to `settings_from_runtime`'s own defaults to silently decide: hub
    has no verified perception/lab runtime dependency yet (no DSN/HTTP
    egress vetted for its event loop), so wiring those is a follow-up, not
    an accident of a missing attr. Weather and the runtime probe (which
    model is currently serving `chat`) ARE enabled -- weather now reads
    orion-hub's own ORION_SITUATION_WEATHER_* fields (added alongside this
    adapter's weather wiring; same provider/coordinates/TTL as cortex-exec's
    already-configured values), and the runtime probe reuses
    `HUB_LLM_GATEWAY_URL`, a host orion-hub already calls today (see
    `/api/llm-routes`). Both `_build_environment_context` and
    `_build_runtime_context` await their blocking `urlopen` calls via
    `asyncio.to_thread` so a cache-miss fetch cannot stall the event loop.

    Affect (2026-08-25) IS enabled here, unlike perception/lab -- orion-hub
    is the MOST verified host for it, not the least: Hub owns the capture
    loop that produces the read in the first place
    (`services/orion-hub/scripts/vision_affect_ambient.py`) and already
    holds a connected bus (`bind_juniper_affect_state_bus` is called at
    startup in `services/orion-hub/scripts/main.py`). No new DSN/HTTP
    egress is needed -- `juniper_affect_state.py`'s read side is a plain
    Redis GET on the bus connection this process already owns.
    """
    return SimpleNamespace(
        orion_situation_enabled=bool(getattr(cfg, "ORION_SITUATION_ENABLED", True)),
        orion_situation_ttl_seconds=int(getattr(cfg, "ORION_SITUATION_TTL_SECONDS", 300)),
        orion_situation_prompt_max_chars=1200,
        orion_situation_timezone=str(getattr(cfg, "ORION_SITUATION_TIMEZONE", "America/Denver")),
        orion_situation_location_label="Unknown",
        orion_situation_locality=None,
        orion_situation_region=None,
        orion_situation_country=None,
        orion_situation_location_precision="city",
        orion_situation_weather_enabled=bool(getattr(cfg, "ORION_SITUATION_WEATHER_ENABLED", True)),
        orion_situation_weather_provider=str(getattr(cfg, "ORION_SITUATION_WEATHER_PROVIDER", "stub")),
        orion_situation_weather_lat=getattr(cfg, "ORION_SITUATION_WEATHER_LAT", None),
        orion_situation_weather_lon=getattr(cfg, "ORION_SITUATION_WEATHER_LON", None),
        orion_situation_weather_ttl_seconds=int(getattr(cfg, "ORION_SITUATION_WEATHER_TTL_SECONDS", 600)),
        # Practical-flag thresholds are shared policy constants, not
        # location data -- cortex-exec's own defaults, not worth a separate
        # hub env key each until an operator actually wants them to diverge.
        orion_situation_umbrella_precip_prob_threshold=40,
        orion_situation_jacket_temp_f_threshold=55,
        orion_situation_high_wind_mph_threshold=25,
        orion_situation_hot_car_temp_f_threshold=80,
        orion_situation_agenda_enabled=False,
        orion_situation_lab_context_enabled=False,
        orion_situation_lab_provider="stub",
        orion_situation_perception_enabled=False,
        orion_situation_perception_max_age_seconds=900,
        orion_situation_perception_stream_id="cam0",
        orion_situation_identity_ask_cooldown_seconds=1200,
        orion_situation_affect_enabled=bool(getattr(cfg, "ORION_SITUATION_AFFECT_ENABLED", True)),
        orion_situation_affect_max_age_seconds=int(
            getattr(cfg, "ORION_SITUATION_AFFECT_MAX_AGE_SECONDS", 300)
        ),
        orion_situation_runtime_enabled=True,
        orion_situation_runtime_route="chat",
        orion_situation_runtime_ttl_seconds=120,
        orion_situation_runtime_probe_timeout_sec=2.0,
        cortex_exec_llm_gateway_url=str(getattr(cfg, "HUB_LLM_GATEWAY_URL", "http://127.0.0.1:8210")),
        orion_presence_default_requestor=str(getattr(cfg, "ORION_PRESENCE_DEFAULT_REQUESTOR", "Juniper")),
        orion_presence_persist_allowed=bool(getattr(cfg, "ORION_PRESENCE_PERSIST_ALLOWED", False)),
    )


def _presence_cache_fingerprint(ctx: dict[str, Any], cfg: SituationSettings) -> str:
    """Fingerprint of the presence-relevant fields that should bust the
    situation cache when they meaningfully change.

    MUST be idempotent under executor.py's own `ctx["presence_context"] =
    situation_brief.get("presence")` self-mutation (executor.py
    ~3029, `call_step_services`), which round-trips this function's OWN
    prior *output* back in as if it were fresh caller *input* on the
    second+ service iteration of the same step within one turn. That means
    every default this function applies has to match `_presence_from_ctx`'s
    defaults byte-for-byte -- fingerprint(raw_caller_input) must equal
    fingerprint(that_same_input_after_one_round_trip_through_the_defaults),
    or a same-turn re-entry within the cache's own TTL window "changes"
    fingerprint purely because a None got filled in with its default value,
    never actually hitting the cache. Confirmed live: an earlier version of
    this function used `raw.get(...)` with no defaulting (or a defaulting
    scheme that didn't match `_presence_from_ctx`), so requestor and
    privacy_mode alone differed enough to bust the cache on literally every
    call within a turn. Consequence when the cache never hits:
    _build_conversation_phase runs again, reads back the
    last_user_turn_at the FIRST call in this turn just wrote to Redis (now
    == "a moment ago"), computes delta_user≈0, and silently reclassifies a
    genuine long_gap/stale_thread as same_breath mid-turn -- exactly the
    kind of masking this whole patch exists to prevent. submitted_at/
    expires_at are excluded entirely rather than defaulted-and-matched:
    they carry microsecond precision and no real caller ever sets them on
    genuine input, so there is no meaningful signal to preserve there.
    """
    raw = ctx.get("presence_context") if isinstance(ctx.get("presence_context"), dict) else {}
    companions = raw.get("companions") if isinstance(raw.get("companions"), list) else []
    normalized_companions = [
        {
            "display_name": item.get("display_name"),
            "relationship": item.get("relationship"),
            "role": item.get("role"),
            "age_band": item.get("age_band"),
        }
        for item in companions[:8]
        if isinstance(item, dict) and item.get("display_name")
    ]
    requestor = raw.get("requestor") if isinstance(raw.get("requestor"), dict) else {}
    # Same default logic as _presence_from_ctx below, field for field.
    audience_mode = str(raw.get("audience_mode") or ("solo" if not normalized_companions else "mixed_group"))
    fingerprint = {
        "audience_mode": audience_mode,
        "companions": normalized_companions,
        "requestor": {
            "display_name": str(requestor.get("display_name") or cfg.default_requestor),
            "relationship_to_orion": str(requestor.get("relationship_to_orion") or "primary_operator"),
        },
        "privacy_mode": str(raw.get("privacy_mode") or "session_only"),
    }
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def _situation_cache_key(ctx: dict[str, Any], cfg: SituationSettings) -> str:
    session_key = str(ctx.get("session_id") or "global")
    # input_modality is part of the key, not just part of the brief.
    #
    # The brief is cached per session for ttl_seconds (300 default), so
    # without this a spoken turn's cached brief is replayed for the next
    # turn even if that one was TYPED -- and the prompt would tell Orion
    # "Juniper SPOKE this turn aloud ... read through the transcription
    # artifacts" about a sentence she typed. The reverse order suppresses
    # the line on a genuinely spoken turn.
    #
    # This was invisible before 2026-08-26 because the value was a
    # constant "typed" on every unified turn (nothing ever supplied
    # surface_context), so the key never needed to distinguish it. Making
    # the field observable is what made the key wrong -- a real review
    # finding, not a hypothetical.
    modality = _build_surface_context(ctx).input_modality
    return f"{session_key}:{modality}:{_presence_cache_fingerprint(ctx, cfg)}"


async def build_situation_for_ctx(ctx: dict[str, Any], runtime_settings: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = settings_from_runtime(runtime_settings)
    if not cfg.enabled:
        return {}, {}
    cache_key = _situation_cache_key(ctx, cfg)
    with _LOCK:
        cached = _SITUATION_CACHE.get(cache_key)
        if cached and (datetime.now(timezone.utc) - cached[0]).total_seconds() < cfg.ttl_seconds:
            return cached[1].model_dump(mode="json"), cached[2].model_dump(mode="json")

    now = datetime.now(timezone.utc)
    diagnostics = SituationDiagnosticsV1()
    presence = _presence_from_ctx(ctx, cfg, now)
    time_ctx = _build_time_context(cfg, diagnostics)
    phase_ctx = await _build_conversation_phase(ctx, time_ctx, now)
    place_ctx = _build_place_context(cfg)
    env_ctx = await _build_environment_context(cfg, diagnostics)
    agenda_ctx = AgendaContextV1(available=False, source="stub")
    lab_ctx = _build_lab_context(cfg)
    perception_ctx = await _build_perception_context(cfg, diagnostics)
    affect_ctx = await _build_affect_context(cfg, diagnostics)
    runtime_ctx = await _build_runtime_context(cfg, diagnostics)
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
            "perception": perception_ctx.source,
            "affect": affect_ctx.source,
            "runtime": runtime_ctx.source,
        },
        requestor=presence.requestor,
        presence=presence,
        time=time_ctx,
        conversation_phase=phase_ctx,
        place=place_ctx,
        environment=env_ctx,
        agenda=agenda_ctx,
        lab=lab_ctx,
        perception=perception_ctx,
        affect=affect_ctx,
        runtime=runtime_ctx,
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


async def _build_conversation_phase(ctx: dict[str, Any], time_ctx: TimeContextV1, now_utc: datetime) -> ConversationPhaseContextV1:
    session_id = str(ctx.get("session_id") or "global")
    state = await read_session_turn_state(session_id)
    last_user = state.last_user_turn_at
    last_orion = state.last_orion_turn_at
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
    # Read-modify-write: preserve last_orion_turn_at exactly as read above --
    # this call only ever advances the user side of the pair. Skip the write
    # entirely if the read itself failed (state.ok is False): last_orion is
    # None in that case because it's UNKNOWN, not because it's genuinely
    # empty, and writing it back would silently clobber a real value on a
    # field this call never intended to touch. Losing this turn's
    # last_user_turn_at update is an acceptable, strictly-better-than-before
    # degradation -- clobbering last_orion_turn_at would not be.
    if state.ok:
        await write_session_turn_state(
            session_id,
            last_user_turn_at=now_utc,
            last_orion_turn_at=last_orion,
        )
    return out


async def mark_orion_turn(session_id: str | None) -> None:
    sid = str(session_id or "global")
    state = await read_session_turn_state(sid)
    if not state.ok:
        # Same clobber hazard as _build_conversation_phase above, mirrored:
        # skip the write rather than risk overwriting a real
        # last_user_turn_at with an unknown-vs-empty None.
        return
    # Read-modify-write: preserve last_user_turn_at exactly as read above --
    # this call only ever advances the Orion side of the pair.
    await write_session_turn_state(
        sid,
        last_user_turn_at=state.last_user_turn_at,
        last_orion_turn_at=datetime.now(timezone.utc),
    )


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


async def _build_environment_context(cfg: SituationSettings, diagnostics: SituationDiagnosticsV1) -> EnvironmentContextV1:
    if not cfg.weather_enabled:
        diagnostics.provider_status["weather"] = "disabled"
        return EnvironmentContextV1(available=False, source="disabled")
    cache_key = f"{cfg.weather_provider}:{cfg.weather_lat}:{cfg.weather_lon}"
    with _LOCK:
        cached = _WEATHER_CACHE.get(cache_key)
        if cached and (datetime.now(timezone.utc) - cached[0]).total_seconds() < cfg.weather_ttl_seconds:
            return cached[1]
    try:
        # `_fetch_weather` is a plain blocking `urlopen` call -- offloaded to a
        # thread for the same reason `_build_runtime_context` already is:
        # this now runs inside orion-hub's single shared WebSocket event loop
        # (execute_unified_turn awaits build_situation_for_ctx directly), not
        # just cortex-exec's per-turn dispatch, so a cache-miss call here must
        # not stall every other concurrent client's turn.
        env = await asyncio.to_thread(_fetch_weather, cfg)
        with _LOCK:
            _WEATHER_CACHE[cache_key] = (datetime.now(timezone.utc), env)
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


def _fetch_runtime_context(cfg: SituationSettings) -> RuntimeContextV1:
    """Live read of what model is actually serving `cfg.runtime_route`.

    Hits orion-llm-gateway's GET /routes (already health-cached there 15s;
    see route_catalog.py's `_probe_model`) rather than probing the backend
    directly -- the gateway already owns route->backend resolution, so this
    reuses that instead of re-deriving it. Mirrors `_fetch_weather`'s shape:
    a plain urlopen with a short timeout, raising on any failure so the
    caller's try/except degrades to unavailable rather than partial/guessed
    data.
    """
    url = f"{cfg.llm_gateway_base_url.rstrip('/')}/routes"
    with urlopen(url, timeout=cfg.runtime_probe_timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(routes, list):
        raise ValueError("routes payload missing/malformed")
    entry = next((r for r in routes if isinstance(r, dict) and r.get("id") == cfg.runtime_route), None)
    if entry is None:
        raise ValueError(f"route {cfg.runtime_route!r} not in /routes response")
    model_id = entry.get("model")
    return RuntimeContextV1(
        available=bool(entry.get("status") == "up" and isinstance(model_id, str) and model_id.strip()),
        route=cfg.runtime_route,
        model_id=model_id if isinstance(model_id, str) and model_id.strip() else None,
        served_by=entry.get("served_by"),
        backend=entry.get("backend"),
        source="orion-llm-gateway",
    )


async def _build_runtime_context(cfg: SituationSettings, diagnostics: SituationDiagnosticsV1) -> RuntimeContextV1:
    if not cfg.runtime_enabled:
        diagnostics.provider_status["runtime"] = "disabled"
        return RuntimeContextV1(available=False, route=cfg.runtime_route, source="disabled")
    cache_key = cfg.runtime_route
    with _LOCK:
        cached = _RUNTIME_CACHE.get(cache_key)
        if cached and (datetime.now(timezone.utc) - cached[0]).total_seconds() < cfg.runtime_ttl_seconds:
            return cached[1]
    try:
        # `_fetch_runtime_context` is a plain blocking `urlopen` call (up to
        # `runtime_probe_timeout_sec`). Offloaded to a thread rather than
        # called inline -- this function runs inside `build_situation_for_ctx`,
        # which orion-hub's `execute_unified_turn` now awaits directly on its
        # single shared event loop (unlike cortex-exec, which already
        # dedicates a worker per chat turn). A cache-miss call here must not
        # stall every other concurrent WebSocket client's turn.
        runtime_ctx = await asyncio.to_thread(_fetch_runtime_context, cfg)
        with _LOCK:
            _RUNTIME_CACHE[cache_key] = (datetime.now(timezone.utc), runtime_ctx)
        diagnostics.provider_status["runtime"] = "ok" if runtime_ctx.available else "unavailable"
        return runtime_ctx
    except Exception as exc:
        diagnostics.provider_status["runtime"] = "error"
        diagnostics.provider_errors["runtime"] = str(exc)
        return RuntimeContextV1(available=False, route=cfg.runtime_route, source="error")


async def _build_perception_context(
    cfg: SituationSettings, diagnostics: SituationDiagnosticsV1
) -> PerceptionContextV1:
    """Most recent camera percept, gated hard on age.

    The staleness gate is the point, not a safety rail bolted on: a percept from
    two hours ago rendered as a current observation is a confabulation with a
    real referent, which is worse than saying nothing. Past the threshold this
    returns `available=False` and the prompt line becomes an explicit "haven't
    seen anything recently".

    Fail-open like every other provider here -- a database problem yields no
    percept, never an exception into turn assembly.

    `async` since 2026-08-26: the identity-uncertain cooldown check below is
    a Redis round-trip (`identity_ask_cooldown.py`), the one await in this
    function -- everything else here stays the same synchronous SQLAlchemy
    reads it always was.
    """
    if not cfg.perception_enabled:
        diagnostics.provider_status["perception"] = "disabled"
        return PerceptionContextV1(available=False, source="disabled")

    try:
        percept = fetch_latest_percept()
    except Exception as exc:  # noqa: BLE001 -- provider contract is fail-open
        diagnostics.provider_status["perception"] = "error"
        diagnostics.provider_errors["perception"] = str(exc)
        return PerceptionContextV1(available=False, source="error")

    if not percept or not percept.get("scene_summary"):
        diagnostics.provider_status["perception"] = "empty"
        return PerceptionContextV1(available=False, source="unavailable")

    age = percept_age_seconds(percept.get("observed_at"))
    if age is None or age > cfg.perception_max_age_seconds:
        diagnostics.provider_status["perception"] = "stale"
        # observed_at is carried even when stale so a debug surface can say how
        # old, but scene_summary is NOT -- a stale summary in the payload is a
        # stale summary one refactor away from reaching a prompt.
        return PerceptionContextV1(
            available=False,
            source="stale",
            observed_at=percept.get("observed_at"),
            observation_age_seconds=age,
        )

    diagnostics.provider_status["perception"] = "ok"
    scene_summary = percept["scene_summary"]
    presence_state = presence_since_sec = presence_subject = None
    presence_identity_uncertain = False

    # Presence is an ENRICHMENT of an already-valid percept, not an
    # independent availability path -- it only folds in when the narrative
    # above already cleared the staleness gate. A room that has not been
    # seen recently should not have "someone was there three hours ago"
    # surface as if it were current.
    presence = fetch_presence(cfg.perception_stream_id)
    if presence:
        presence_state = presence.get("state")
        presence_since_sec = presence.get("since_sec")
        presence_subject = presence.get("subject")
        fragment = presence_fragment(presence_state, presence_since_sec)
        if fragment:
            scene_summary = f"{fragment} {scene_summary}"
            diagnostics.provider_status["perception_presence"] = presence_state or "unknown"

        # identity_uncertain reaching True here means: a real person is
        # believed present right now AND identity_face genuinely did not
        # match (see presence.py's own staleness/stickiness rules -- this is
        # already the freshest, sticky-against-flicker read). The cooldown
        # claim is the only thing standing between that and asking every
        # single turn for as long as the mismatch persists -- see
        # identity_ask_cooldown.py's module docstring for why an in-process
        # flag would repeat a bug this codebase already fixed once, and why
        # the claim is a SINGLE atomic call (review finding, 2026-08-26: a
        # separate check-then-set let two concurrent cortex-exec replicas
        # both read "not in cooldown" and both ask).
        #
        # No try/except here, matching this function's own established
        # convention for fetch_presence above: try_claim_identity_ask is
        # itself documented "never raises" (fail-open internally, logging a
        # warning instead) -- double-wrapping an already-fail-open callee is
        # ceremony this file doesn't otherwise carry, not extra safety.
        if presence.get("identity_uncertain") and await try_claim_identity_ask(
            cfg.perception_stream_id, ttl_seconds=cfg.identity_ask_cooldown_seconds
        ):
            presence_identity_uncertain = True

    return PerceptionContextV1(
        available=True,
        source="live",
        scene_summary=scene_summary,
        observed_at=percept.get("observed_at"),
        observation_age_seconds=age,
        stream_id=cfg.perception_stream_id,
        presence_state=presence_state,
        presence_since_sec=presence_since_sec,
        presence_subject=presence_subject,
        presence_identity_uncertain=presence_identity_uncertain,
    )


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


async def _build_affect_context(
    cfg: SituationSettings, diagnostics: SituationDiagnosticsV1
) -> AffectContextV1:
    """Most recent facial+vocal affect read of Juniper, gated hard on age.

    Same staleness-gate reasoning `_build_perception_context` documents: an
    affect read from 20 minutes ago rendered as current is a confabulation
    with a real referent, worse than saying nothing. Past the threshold this
    returns `available=False` and the prompt line becomes an explicit "no
    recent affect read" -- never a stale mood presented as current.

    Fail-open like every other provider here -- a Redis problem yields no
    affect read, never an exception into turn assembly.
    """
    if not cfg.affect_enabled:
        diagnostics.provider_status["affect"] = "disabled"
        return AffectContextV1(available=False, source="disabled")

    try:
        state = await read_latest_juniper_affect()
    except Exception as exc:  # noqa: BLE001 -- provider contract is fail-open
        diagnostics.provider_status["affect"] = "error"
        diagnostics.provider_errors["affect"] = str(exc)
        return AffectContextV1(available=False, source="error")

    if not state.ok:
        diagnostics.provider_status["affect"] = "error"
        return AffectContextV1(available=False, source="error")

    if not state.summary or not state.observed_at:
        diagnostics.provider_status["affect"] = "empty"
        return AffectContextV1(available=False, source="unavailable")

    age = percept_age_seconds(state.observed_at)
    if age is None or age > cfg.affect_max_age_seconds:
        diagnostics.provider_status["affect"] = "stale"
        # observed_at is carried even when stale so a debug surface can say
        # how old, but summary is NOT -- a stale mood read in the payload is
        # a stale read one refactor away from reaching a prompt.
        return AffectContextV1(
            available=False,
            source="stale",
            observed_at=state.observed_at,
            observation_age_seconds=age,
        )

    diagnostics.provider_status["affect"] = "ok"
    return AffectContextV1(
        available=True,
        summary=state.summary,
        observed_at=state.observed_at,
        observation_age_seconds=age,
        trigger=state.trigger,
        subtitle_source=state.subtitle_source,
        source="live",
    )


def _build_prompt_fragment(brief: SituationBriefV1, max_chars: int) -> SituationPromptFragmentV1:
    lines = [
        f"Local context: {brief.time.time_of_day_label.replace('_', ' ')} {brief.time.weekday}, {brief.time.timezone}.",
        f"Conversation phase: {brief.conversation_phase.phase_change}; continuity={brief.conversation_phase.continuity_mode}.",
        f"Presence: requestor={brief.requestor.display_name}, audience_mode={brief.presence.audience_mode}.",
    ]
    # Only rendered for a non-typed modality. SurfaceContextV1.input_modality
    # has existed since this brief was first built, but nothing ever put it
    # in the prompt -- a schema field with no consumer. It earns a line here
    # because the answer changes behaviour: a spoken turn was dictated, so
    # homophones and run-on phrasing are transcription artifacts rather than
    # things Juniper wrote, and Orion should not read into them. "typed"
    # stays silent rather than emitting a line on every single turn to say
    # nothing happened -- same only-when-it-means-something discipline the
    # relevance/affordance lines below already follow.
    if brief.surface.input_modality == "spoken":
        # Kept deliberately short. An earlier 246-char version of this line
        # pushed the fragment into the cap and cost a caution line; the
        # budget here is shared, so verbosity in one line is silence in
        # another.
        lines.append(
            "Input modality: Juniper SPOKE this turn aloud (Whisper "
            "transcript) -- odd wording and missing punctuation are "
            "transcription artifacts, not word choice."
        )
    elif brief.surface.input_modality not in ("typed", "unknown"):
        lines.append(f"Input modality: {brief.surface.input_modality}.")
    if brief.environment.available:
        rain = brief.environment.forecast_next_6h.precipitation_probability_pct
        lines.append(f"Weather next 6h: precip_prob={rain}%, summary={brief.environment.forecast_next_6h.summary}.")
    else:
        lines.append("Weather: unavailable or low-confidence; do not infer.")
    if brief.lab.available:
        lines.append(f"Lab risk: thermal={brief.lab.thermal_risk}, power={brief.lab.power_risk}.")
    else:
        lines.append("Lab: unavailable/stub; do not infer.")
    if brief.perception.available and brief.perception.scene_summary:
        age_min = round((brief.perception.observation_age_seconds or 0) / 60)
        seen = "just now" if age_min < 1 else f"{age_min} min ago"
        lines.append(f"Room (seen {seen}): {brief.perception.scene_summary}")
    else:
        # Never phrase this as "the room is empty/quiet" -- not seeing and
        # seeing nothing are different claims, and only one of them is true.
        lines.append("Room: haven't seen anything recently; do not infer.")
    if brief.affect.available and brief.affect.summary:
        age_min = round((brief.affect.observation_age_seconds or 0) / 60)
        seen = "just now" if age_min < 1 else f"{age_min} min ago"
        no_voice = " (no speech detected)" if brief.affect.subtitle_source == "none" else ""
        lines.append(
            f"Juniper's affect (captured {seen}{no_voice}): {brief.affect.summary}"
        )
    else:
        # Same honesty rule as Room above -- no recent capture and a
        # deliberately-not-captured mood are different claims.
        lines.append("Juniper's affect: no recent capture; do not infer.")
    if brief.runtime.available and brief.runtime.model_id:
        lines.append(f"You are currently running on model: {brief.runtime.model_id} (route={brief.runtime.route}).")
    else:
        lines.append("Current model: unavailable; do not infer or guess a name.")
    relevance = [f"{a.kind}: {a.suggestion}" for a in brief.affordances if a.trigger_relevance == "active"]
    # Ordered most-important-first: whatever the budget cannot fit is
    # dropped from the END (see the append loop below). The affect guard
    # leads because it is the only one of the three whose absence changes
    # how Orion may talk about a real reading of Juniper's face and voice;
    # the other two are style guidance.
    cautions = [
        "Juniper's affect read is a model's inference, not a diagnosis or a fixed label -- treat it as one signal, not a certainty, and don't announce it unprompted.",
        "Situation context is grounding, not a requirement to mention.",
        "Use only when relevant; avoid contrived time/weather/location commentary.",
    ]
    if brief.perception.presence_identity_uncertain:
        # Inserted first: this is the one caution that's actionable THIS
        # turn and time-limited (a fresh identity_face mismatch, already
        # cooldown-gated server-side so this only ever appears once per
        # sit-down -- see identity_ask_cooldown.py). Losing it to truncation
        # would silently drop the one thing Juniper explicitly asked Orion
        # to do here.
        cautions.insert(
            0,
            "You don't recognize the person currently in view with confidence "
            "(a camera identity check came back unsure, not a name match). If it "
            "feels natural in this turn, ask ONE brief, warm clarifying question -- "
            "e.g. \"Hi, I'm having a little trouble recognizing you -- is that you, "
            "Juniper?\" This has not been asked recently for this camera, so it is "
            "safe to ask now; do not repeat it again this conversation once asked.",
        )
    # The cautions are appended AFTER truncation, never inside it.
    #
    # This used to be one flat join sliced from the tail, which meant the
    # cautions -- last in the list -- were the first thing the cap ate.
    # Confirmed live 2026-08-26: with a 300-char affect summary (the real
    # _AFFECT_SUMMARY_MAX_CHARS ceiling) plus the spoken-modality line, the
    # fragment hit exactly 1200 and cut
    # "...treat it as one signal, not a certainty, and don't announce it
    # unprompted." off mid-sentence -- i.e. the privacy guard was dropped
    # from precisely the turn on which a webcam capture had just been fired
    # at Juniper. Losing grounding detail to a cap is acceptable; losing the
    # instruction about how to handle that detail is not.
    #
    # So: body (facts, which are safe to shorten) is truncated to whatever
    # room is left once the cautions are reserved; cautions are then always
    # appended in full.
    body = "Situation:\n- " + "\n- ".join(lines + relevance)
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    # Cautions are appended WHOLE or not at all -- never sliced.
    #
    # This used to be one flat join sliced from the tail, so the cautions
    # (last in the list) were the first thing the cap ate, mid-sentence.
    # Confirmed live 2026-08-26 at the production 1200 cap: a 300-char
    # affect summary plus the spoken-modality line cut
    # "...treat it as one signal, not a certainty, and don't announce it
    # unprompted." in half -- dropping the guard on how to handle Juniper's
    # affect read from exactly the turn on which a capture had been fired at
    # her. Losing grounding facts to a cap is acceptable; emitting a
    # half-sentence instruction is not, and silently losing the privacy
    # guard is worse.
    #
    # Priority order matters: cautions are emitted most-important-first, so
    # if the budget only fits some, the ones that survive are the ones whose
    # absence would actually change behaviour. Body keeps first claim on the
    # budget, which is why an artificially small cap (the 400 used in
    # test_situation_provider's fixture, vs 1200 in production) still
    # produces a usable brief rather than nothing but boilerplate.
    remaining = max_chars - len(body)
    kept: list[str] = []
    for caution in cautions:
        cost = len(caution) + len("\n- ")
        if cost <= remaining:
            kept.append(caution)
            remaining -= cost
    compact = body + ("".join("\n- " + c for c in kept) if kept else "")
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
