from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import orion.situational.context as situation
from orion.situational.session_turn_phase import (
    bind_session_turn_phase_bus,
    reset_session_turn_phase_bus_for_tests,
    write_session_turn_state,
)
from orion.situational.context import build_situation_for_ctx


def _settings(**overrides):
    base = {
        "orion_situation_enabled": True,
        "orion_situation_ttl_seconds": 300,
        "orion_situation_prompt_max_chars": 400,
        "orion_situation_timezone": "America/Denver",
        "orion_situation_location_label": "Utah",
        "orion_situation_locality": "Vernal",
        "orion_situation_region": "Utah",
        "orion_situation_country": "US",
        "orion_situation_location_precision": "city",
        "orion_situation_weather_enabled": False,
        "orion_situation_weather_provider": "stub",
        "orion_situation_weather_lat": None,
        "orion_situation_weather_lon": None,
        "orion_situation_weather_ttl_seconds": 600,
        "orion_situation_umbrella_precip_prob_threshold": 40,
        "orion_situation_jacket_temp_f_threshold": 55,
        "orion_situation_high_wind_mph_threshold": 25,
        "orion_situation_hot_car_temp_f_threshold": 80,
        "orion_situation_agenda_enabled": False,
        "orion_situation_lab_context_enabled": True,
        "orion_situation_lab_provider": "stub",
        # False here (unlike the True production default) for the same
        # reason weather_enabled is False above: avoid a real network call
        # to orion-llm-gateway from every unrelated situation test.
        "orion_situation_runtime_enabled": False,
        "orion_situation_runtime_route": "chat",
        "orion_situation_runtime_ttl_seconds": 120,
        "orion_situation_runtime_probe_timeout_sec": 2.0,
        "orion_presence_default_requestor": "Juniper",
        "orion_presence_persist_allowed": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeSessionTurnPhaseRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        self.store[key] = payload.encode("utf-8")


class _FakeSessionTurnPhaseBus:
    def __init__(self, redis: _FakeSessionTurnPhaseRedis) -> None:
        self.redis = redis


# NOTE: build_situation_for_ctx (and, inside it, _build_conversation_phase)
# is async since conversation-phase turn timestamps moved to Redis
# (session_turn_phase.py) -- see that module's docstring for why. No bus is
# bound in these tests, so the Redis read/write inside it fails open (a
# "bus_unbound" WARNING is expected and harmless) and phase resolves to its
# safe default ("unknown"/"continue_directly"), same as a fresh session with
# no prior state. These tests only assert on weather/presence/runtime/
# affordance behavior, not on phase-bucketing itself (that's covered by
# test_session_turn_phase.py).


@pytest.mark.asyncio
async def test_situation_marks_temporal_resume_for_long_gap():
    ctx = {"session_id": "sid-temporal", "raw_user_text": "yeah do that"}
    settings = _settings()
    await build_situation_for_ctx(ctx, settings)
    brief, fragment = await build_situation_for_ctx(ctx, settings)
    assert brief["kind"] == "situation.brief.v1"
    assert "conversation_phase" in brief
    assert fragment["kind"] == "situation.prompt_fragment.v1"


@pytest.mark.asyncio
async def test_situation_presence_child_affordance():
    from orion.situational import context as situation_mod

    situation_mod._SITUATION_CACHE.clear()
    ctx = {
        "session_id": "sid-kid",
        "raw_user_text": "can you explain this for my kid",
        "presence_context": {
            "audience_mode": "kid_present",
            "companions": [{"display_name": "Kid", "relationship": "child", "role": "asker", "age_band": "child"}],
            "requestor": {"display_name": "Juniper"},
        },
    }
    brief, fragment = await build_situation_for_ctx(ctx, _settings())
    kinds = {item["kind"] for item in brief["affordances"]}
    assert brief["presence"]["audience_mode"] == "kid_present"
    assert "kid_friendly_explanation" in kinds
    assert "kid_present" in fragment["compact_text"]


@pytest.mark.asyncio
async def test_situation_cache_refreshes_when_presence_changes():
    from orion.situational import context as situation_mod

    situation_mod._SITUATION_CACHE.clear()
    session = "sid-cache-presence"
    brief_solo, _ = await build_situation_for_ctx(
        {"session_id": session, "presence_context": {"audience_mode": "solo"}},
        _settings(orion_situation_ttl_seconds=300),
    )
    brief_kid, fragment_kid = await build_situation_for_ctx(
        {
            "session_id": session,
            "presence_context": {
                "audience_mode": "kid_present",
                "companions": [{"display_name": "Kid", "relationship": "child", "role": "listener", "age_band": "child"}],
            },
        },
        _settings(orion_situation_ttl_seconds=300),
    )
    assert brief_solo["presence"]["audience_mode"] == "solo"
    assert brief_kid["presence"]["audience_mode"] == "kid_present"
    assert "kid_present" in fragment_kid["compact_text"]


@pytest.mark.asyncio
async def test_situation_cache_refreshes_when_requestor_changes():
    from orion.situational import context as situation_mod

    situation_mod._SITUATION_CACHE.clear()
    session = "sid-cache-requestor"
    await build_situation_for_ctx(
        {
            "session_id": session,
            "presence_context": {
                "audience_mode": "solo",
                "requestor": {"display_name": "Juniper"},
            },
        },
        _settings(orion_situation_ttl_seconds=300),
    )
    brief_guest, _ = await build_situation_for_ctx(
        {
            "session_id": session,
            "presence_context": {
                "audience_mode": "solo",
                "requestor": {"display_name": "Guest"},
            },
        },
        _settings(orion_situation_ttl_seconds=300),
    )
    assert brief_guest["presence"]["requestor"]["display_name"] == "Guest"


@pytest.mark.asyncio
async def test_situation_cache_survives_executors_own_presence_context_mutation():
    """Regression test for a real bug found in review (2026-08-21): the
    cache fingerprint used to include raw.get("submitted_at")/("expires_at"),
    which are always None on genuine caller input but get populated once
    executor.py's own `ctx["presence_context"] = situation_brief.get
    ("presence")` (executor.py, `call_step_services`) round-trips this
    function's own prior output back in as if it were fresh input, on the
    second+ service iteration of the SAME step. expires_at carries
    microsecond precision, so that self-mutation changed the cache key on
    essentially every call -- meaning a same-turn re-entry within the
    _SITUATION_CACHE's own TTL window never actually hit the cache.

    Consequence when it doesn't hit: _build_conversation_phase runs again,
    reads back the last_user_turn_at the FIRST call in this turn just wrote
    to Redis (now == "a moment ago"), computes delta_user ≈ 0, and silently
    reclassifies a genuine long_gap as same_breath -- overwriting the
    correct phase mid-turn with exactly the kind of masking this whole
    patch exists to prevent. This test seeds a real long_gap timestamp,
    calls build_situation_for_ctx once, applies the exact ctx mutation
    executor.py performs, then calls it again immediately -- the second
    call must return the SAME cached long_gap brief, not a fresh
    same_breath one.
    """
    reset_session_turn_phase_bus_for_tests()
    bind_session_turn_phase_bus(_FakeSessionTurnPhaseBus(_FakeSessionTurnPhaseRedis()))
    try:
        situation._SITUATION_CACHE.clear()
        session = "sid-mutation-regression"
        six_hours_ago = datetime.now(timezone.utc) - timedelta(hours=6)
        await write_session_turn_state(session, last_user_turn_at=six_hours_ago, last_orion_turn_at=None)

        ctx = {"session_id": session, "presence_context": {"audience_mode": "solo"}}
        cfg = _settings(orion_situation_ttl_seconds=300)

        brief_first, _ = await build_situation_for_ctx(ctx, cfg)
        phase_first = brief_first["conversation_phase"]["phase_change"]
        # Not asserting the exact bucket name: _build_conversation_phase's
        # separate crossed_day_boundary check (a real, pre-existing, out-of-
        # scope-for-this-patch quirk -- see test_situation_conversation_phase.py)
        # calls the REAL wall-clock datetime.now(), so "6 hours ago" reads
        # as either long_gap or next_day depending on what time this suite
        # happens to run. Either way it must not be same_breath/unknown.
        assert phase_first not in ("same_breath", "unknown", "short_pause")
        seconds_first = brief_first["conversation_phase"]["time_since_last_user_turn_seconds"]
        assert seconds_first is not None and seconds_first > 3600

        # Exactly executor.py's own mutation (executor.py ~3029):
        ctx["presence_context"] = brief_first.get("presence")

        brief_second, _ = await build_situation_for_ctx(ctx, cfg)
        phase_second = brief_second["conversation_phase"]["phase_change"]
        seconds_second = brief_second["conversation_phase"]["time_since_last_user_turn_seconds"]
        assert phase_second == phase_first, (
            "cache miss caused by the executor's own ctx mutation reclassified "
            f"the conversation phase mid-turn: {phase_first!r} -> {phase_second!r}"
        )
        assert seconds_second == seconds_first, (
            "cache miss caused by the executor's own ctx mutation recomputed "
            "time_since_last_user_turn_seconds mid-turn instead of serving the "
            "cached brief -- this is the exact mechanism that would silently "
            "reclassify a genuine long_gap as same_breath"
        )
    finally:
        reset_session_turn_phase_bus_for_tests()


@pytest.mark.asyncio
async def test_situation_outdoor_departure_affordance():
    ctx = {"session_id": "sid-outdoor", "raw_user_text": "I am heading out the door soon"}
    brief, fragment = await build_situation_for_ctx(ctx, _settings())
    kinds = {item["kind"] for item in brief["affordances"]}
    assert "outdoor_departure" in kinds
    assert len(fragment["compact_text"]) <= 400


@pytest.mark.asyncio
async def test_situation_disabled_returns_no_brief_or_fragment():
    ctx = {"session_id": "sid-disabled", "raw_user_text": "hello"}
    brief, fragment = await build_situation_for_ctx(ctx, _settings(orion_situation_enabled=False))
    assert brief == {}
    assert fragment == {}


class _FakeUrlopenResponse:
    """Mimics the `with urlopen(...) as resp: resp.read()` shape `_fetch_weather` uses."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


@pytest.fixture(autouse=True)
def _clear_runtime_cache():
    # _RUNTIME_CACHE keys only by route name ("chat" in every test below),
    # not by session -- unlike _SITUATION_CACHE/_WEATHER_CACHE it would
    # otherwise leak a mocked result across unrelated test cases.
    situation._RUNTIME_CACHE.clear()
    yield
    situation._RUNTIME_CACHE.clear()


def _pool_view(monkeypatch, routes=None, *, source="gpu_pool", fail=None):
    """GPU pool stage 6.3: the runtime line reads the pool's route view, not the gateway's
    GET /routes. Patches that one read; returns a call counter."""
    import orion.situational.runtime_route_view as store

    calls = {"n": 0}

    async def _read(*, timeout_sec):
        calls["n"] += 1
        if fail is not None:
            raise fail
        return {"source": source, "routes": list(routes or [])}

    monkeypatch.setattr(store, "read_route_view", _read)
    return calls


def _no_pool_read(monkeypatch):
    return _pool_view(monkeypatch, fail=AssertionError("the pool route view must not be read"))


@pytest.fixture(autouse=True)
def _clear_weather_cache():
    # _WEATHER_CACHE keys by "{provider}:{lat}:{lon}", constant across the
    # weather tests below -- same leak risk as _RUNTIME_CACHE above.
    situation._WEATHER_CACHE.clear()
    yield
    situation._WEATHER_CACHE.clear()


@pytest.mark.asyncio
async def test_weather_context_reports_live_conditions_when_openmeteo_configured(monkeypatch):
    """Regression guard for the 2026-08-23 "get weather" follow-up:
    `_build_environment_context` is now async (its blocking urlopen call
    moved to asyncio.to_thread, same fix already applied to
    `_build_runtime_context`) -- confirm the conversion still produces a
    real, populated EnvironmentContextV1 end to end, not just that it
    doesn't crash."""

    def _urlopen(url, timeout=None):
        return _FakeUrlopenResponse(
            {
                "current": {
                    "temperature_2m": 68.0,
                    "apparent_temperature": 66.0,
                    "weather_code": 1,
                    "wind_speed_10m": 5.0,
                    "wind_gusts_10m": 9.0,
                },
                "hourly": {
                    "precipitation_probability": [10, 12, 15, 20, 25, 30],
                    "temperature_2m": [68, 70, 71, 69, 65, 60],
                    "wind_speed_10m": [5, 6, 7, 8, 9, 10],
                },
            }
        )

    monkeypatch.setattr(situation, "urlopen", _urlopen)
    cfg = situation.settings_from_runtime(
        _settings(
            orion_situation_weather_enabled=True,
            orion_situation_weather_provider="openmeteo",
            orion_situation_weather_lat=41.2230,
            orion_situation_weather_lon=-111.9738,
        )
    )
    diagnostics = situation.SituationDiagnosticsV1()
    env = await situation._build_environment_context(cfg, diagnostics)

    assert env.available is True
    assert env.source == "openmeteo"
    assert env.current_weather.temperature_f == 68.0
    assert diagnostics.provider_status["weather"] == "ok"


@pytest.mark.asyncio
async def test_weather_context_caches_within_ttl(monkeypatch):
    calls = {"n": 0}

    def _urlopen(url, timeout=None):
        calls["n"] += 1
        return _FakeUrlopenResponse(
            {
                "current": {"temperature_2m": 55.0, "weather_code": 0},
                "hourly": {"precipitation_probability": [], "temperature_2m": [], "wind_speed_10m": []},
            }
        )

    monkeypatch.setattr(situation, "urlopen", _urlopen)
    cfg = situation.settings_from_runtime(
        _settings(
            orion_situation_weather_enabled=True,
            orion_situation_weather_provider="openmeteo",
            orion_situation_weather_lat=41.2230,
            orion_situation_weather_lon=-111.9738,
        )
    )
    diagnostics = situation.SituationDiagnosticsV1()
    await situation._build_environment_context(cfg, diagnostics)
    await situation._build_environment_context(cfg, diagnostics)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_runtime_context_reports_live_model_when_route_is_up(monkeypatch):
    _pool_view(monkeypatch, [
        {
            "id": "chat",
            "served_by": "circe-worker-1",
            "backend": "llamacpp",
            "status": "up",
            "last_checked_at": "2026-08-14T00:00:00+00:00",
            "model": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
            "role": "chat",
        }
    ])
    ctx = {"session_id": "sid-runtime-up", "raw_user_text": "hello"}
    # prompt_max_chars overridden to the real production default (read from
    # situation._DEFAULT_PROMPT_MAX_CHARS rather than a hardcoded literal --
    # 7200 as of 2026-08-30, was 1200 -- so this can never silently drift
    # from the real value), not the shared _settings() helper's tight 400 --
    # this test is about runtime-model plumbing reaching the prompt, not
    # about truncation ordering, and the shared 400 budget is
    # arbitrary/coincidental here, not a deliberate constraint of this
    # test. 2026-08-25: the affect provider's own "no recent capture" line
    # (~50 chars, always present once affect_enabled defaults True) pushed
    # this specific assertion past 400's truncation boundary -- a real
    # content-budget interaction, not a bug in either provider.
    brief, fragment = await build_situation_for_ctx(
        ctx,
        _settings(
            orion_situation_runtime_enabled=True,
            orion_situation_prompt_max_chars=situation._DEFAULT_PROMPT_MAX_CHARS,
        ),
    )
    assert brief["runtime"]["available"] is True
    assert brief["runtime"]["model_id"] == "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
    assert brief["runtime"]["served_by"] == "circe-worker-1"
    assert brief["runtime"]["source"] == "gpu_pool"
    assert "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf" in fragment["compact_text"]
    # No lease known: the route view is only the route's default under the GPU pool.
    assert brief["runtime"]["placement"] == "route_default"
    assert "Default model for route chat: Qwen3.6-35B-A3B-UD-Q5_K_M.gguf" in fragment["compact_text"]
    assert "You are running on model" not in fragment["compact_text"]


@pytest.mark.asyncio
async def test_runtime_context_spilled_lease_names_granted_role_not_route_default(monkeypatch):
    """GPU pool spill: the turn's hold was granted agent-gpu2. The brief must state the model the
    pool discovered on agent-gpu2, never the route's default -- and must not even read the route
    view (the lease is the fact)."""
    calls = _pool_view(monkeypatch, [
        {"id": "chat", "status": "up", "model": "Route-Default-35B.gguf", "served_by": "circe-worker-chat"},
    ])
    ctx = {
        "session_id": "sid-runtime-spill",
        "raw_user_text": "hello",
        "gpu_placement": {"role": "agent-gpu2", "model": "Spilled-Gpu2-27B.gguf", "profile": "gpu2-flex"},
    }
    brief, fragment = await build_situation_for_ctx(
        ctx,
        _settings(
            orion_situation_runtime_enabled=True,
            orion_situation_prompt_max_chars=situation._DEFAULT_PROMPT_MAX_CHARS,
        ),
    )
    assert calls["n"] == 0
    rt = brief["runtime"]
    assert (rt["placement"], rt["granted_role"], rt["model_id"], rt["profile_name"]) == (
        "lease", "agent-gpu2", "Spilled-Gpu2-27B.gguf", "gpu2-flex"
    )
    assert rt["route"] == "agent"  # the lease route, never the configured runtime_route "chat"
    assert rt["source"] == "gpu_pool_lease"
    text = fragment["compact_text"]
    assert "You are running on model: Spilled-Gpu2-27B.gguf (GPU pool role agent-gpu2, profile gpu2-flex" in text
    assert "Route-Default-35B" not in text


@pytest.mark.asyncio
async def test_runtime_context_lease_without_discovered_model_does_not_guess(monkeypatch):
    _no_pool_read(monkeypatch)
    ctx = {"session_id": "sid-runtime-lease-nomodel", "raw_user_text": "hello",
           "gpu_placement": {"role": "chat", "model": None, "profile": None}}
    brief, fragment = await build_situation_for_ctx(
        ctx,
        _settings(
            orion_situation_runtime_enabled=True,
            orion_situation_prompt_max_chars=situation._DEFAULT_PROMPT_MAX_CHARS,
        ),
    )
    assert brief["runtime"]["available"] is False
    assert brief["runtime"]["granted_role"] == "chat"
    assert "holds the GPU pool's chat role" in fragment["compact_text"]
    assert "do not infer or guess a name" in fragment["compact_text"]


@pytest.mark.asyncio
async def test_runtime_context_mismatch_role_says_why_it_names_no_model(monkeypatch):
    _no_pool_read(monkeypatch)
    ctx = {"session_id": "sid-runtime-mismatch", "raw_user_text": "hello",
           "gpu_placement": {"role": "fast", "model": None, "profile": None, "status": "mismatch"}}
    _, fragment = await build_situation_for_ctx(
        ctx, _settings(orion_situation_runtime_enabled=True,
                       orion_situation_prompt_max_chars=situation._DEFAULT_PROMPT_MAX_CHARS))
    assert "the pool reports that role as mismatch" in fragment["compact_text"]
    assert "could not be read" not in fragment["compact_text"]


@pytest.mark.asyncio
async def test_runtime_line_is_omitted_when_the_harness_owns_it(monkeypatch):
    """Unheld unified turn: the harness prompt names its own route's default, so the brief states
    neither a second route default nor an 'unavailable' placeholder -- and does not read the pool."""
    _no_pool_read(monkeypatch)
    ctx = {"session_id": "sid-runtime-harness", "raw_user_text": "hello", "runtime_line_owner": "harness"}
    brief, fragment = await build_situation_for_ctx(
        ctx, _settings(orion_situation_runtime_enabled=True,
                       orion_situation_prompt_max_chars=situation._DEFAULT_PROMPT_MAX_CHARS))
    assert brief["runtime"]["placement"] == "harness"
    text = fragment["compact_text"]
    assert "Default model for route" not in text
    assert "Current model:" not in text


@pytest.mark.asyncio
async def test_situation_cache_never_replays_another_turns_placement(monkeypatch):
    """The brief is cached per session; the lease is per turn. Two turns in one session on
    different roles must each see their own role."""
    _no_pool_read(monkeypatch)
    cfg = _settings(
        orion_situation_runtime_enabled=True,
        orion_situation_prompt_max_chars=situation._DEFAULT_PROMPT_MAX_CHARS,
    )
    base = {"session_id": "sid-runtime-cache", "raw_user_text": "hello"}
    _, first = await build_situation_for_ctx(
        {**base, "gpu_placement": {"role": "agent", "model": "A.gguf", "profile": "pa"}}, cfg
    )
    _, second = await build_situation_for_ctx(
        {**base, "gpu_placement": {"role": "chat", "model": "C.gguf", "profile": "pc"}}, cfg
    )
    assert "GPU pool role agent" in first["compact_text"]
    assert "GPU pool role chat" in second["compact_text"]
    assert "A.gguf" not in second["compact_text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("unreachable", ["unavailable_view", "raises"])
async def test_runtime_context_degrades_when_pool_unreachable(monkeypatch, unreachable):
    """No pool state (unbound bus, RPC timeout) is the all-unknown view: unavailable, never a
    guessed model, and not cached (the next turn asks again)."""
    if unreachable == "raises":
        _pool_view(monkeypatch, fail=OSError("bus down"))
    else:
        _pool_view(monkeypatch, [{"id": "chat", "status": "unknown", "model": None}],
                   source="gpu_pool_unavailable")
    ctx = {"session_id": f"sid-runtime-down-{unreachable}", "raw_user_text": "hello"}
    brief, fragment = await build_situation_for_ctx(ctx, _settings(orion_situation_runtime_enabled=True))
    assert brief["runtime"]["available"] is False
    assert brief["runtime"]["model_id"] is None
    assert brief["diagnostics"]["provider_status"]["runtime"] == "error"
    assert "unavailable" in fragment["compact_text"].lower()
    assert "Qwen" not in fragment["compact_text"]
    assert situation._RUNTIME_CACHE == {}


@pytest.mark.asyncio
async def test_runtime_context_degrades_when_route_missing_from_response(monkeypatch):
    _pool_view(monkeypatch, [])
    ctx = {"session_id": "sid-runtime-missing-route", "raw_user_text": "hello"}
    brief, _ = await build_situation_for_ctx(ctx, _settings(orion_situation_runtime_enabled=True))
    assert brief["runtime"]["available"] is False
    assert brief["diagnostics"]["provider_status"]["runtime"] == "error"


@pytest.mark.asyncio
async def test_runtime_context_disabled_by_default_in_shared_fixture(monkeypatch):
    # Sanity check on the shared _settings() default itself: it must be
    # False, or every unrelated situation test would attempt a real
    # read of GPU pool state.
    called = _no_pool_read(monkeypatch)
    ctx = {"session_id": "sid-runtime-disabled", "raw_user_text": "hello"}
    brief, fragment = await build_situation_for_ctx(ctx, _settings())
    assert called["n"] == 0
    assert brief["runtime"]["available"] is False
    assert brief["runtime"]["source"] == "disabled"
    assert "unavailable" in fragment["compact_text"].lower()


@pytest.mark.asyncio
async def test_runtime_context_caches_within_ttl(monkeypatch):
    calls = _pool_view(monkeypatch, [
        {"id": "chat", "served_by": "circe-worker-1", "backend": "llamacpp", "status": "up",
         "model": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"}
    ])
    cfg = situation.settings_from_runtime(_settings(orion_situation_runtime_enabled=True))
    diagnostics = situation.SituationDiagnosticsV1()
    await situation._build_runtime_context(cfg, diagnostics)
    await situation._build_runtime_context(cfg, diagnostics)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_runtime_context_reads_pool_state_not_the_gateway(monkeypatch):
    """GPU pool stage 6.3, end to end through the real store and route view: the read is one
    orion:gpu_pool:state RPC with the pool's config, and no HTTP request is made at all."""
    import orion.situational.runtime_route_view as store
    from orion.gpu_pool.config import load_pool_config

    cfg_payload = load_pool_config().model_dump(mode="json", by_alias=True, exclude={"digest"})
    state = {"generated_at": "2026-09-30T00:00:00Z",
             "cards": [{"card": "gpu0", "vram_gb": 32, "lendable": True, "lent": False}],
             "roles": [{"role": "chat", "kind": "llm", "cards": ["gpu0"], "url": "http://h:8011",
                        "status": "confirmed", "model_file": "Chat-35B.gguf",
                        "model_path": "/models/gguf/Chat-35B.gguf", "ctx_per_slot": 131072}],
             "config": cfg_payload}
    sent = []

    class _Decoded:
        ok = True
        envelope = type("E", (), {"payload": state})()

    class _Bus:
        codec = type("C", (), {"decode": lambda self, data: _Decoded()})()

        async def rpc_request(self, channel, env, **kw):
            sent.append((channel, env.payload))
            return {"data": b"raw"}

    def _no_http(url, timeout=None):
        raise AssertionError(f"no HTTP read expected, got {url}")

    monkeypatch.setattr(situation, "urlopen", _no_http)
    monkeypatch.setattr(store, "_BUS", _Bus())
    cfg = situation.settings_from_runtime(_settings(orion_situation_runtime_enabled=True))
    runtime = await situation._build_runtime_context(cfg, situation.SituationDiagnosticsV1())
    assert sent == [("orion:gpu_pool:state:request", {"include_leases": False, "include_config": True})]
    assert (runtime.available, runtime.model_id, runtime.served_by, runtime.source) == (
        True, "/models/gguf/Chat-35B.gguf", "circe-worker-chat", "gpu_pool")


@pytest.mark.asyncio
async def test_runtime_context_unbound_bus_is_unavailable(monkeypatch):
    import orion.situational.runtime_route_view as store

    monkeypatch.setattr(store, "_BUS", None)
    cfg = situation.settings_from_runtime(_settings(orion_situation_runtime_enabled=True))
    diagnostics = situation.SituationDiagnosticsV1()
    runtime = await situation._build_runtime_context(cfg, diagnostics)
    assert runtime.available is False and runtime.model_id is None
    assert diagnostics.provider_status["runtime"] == "error"
