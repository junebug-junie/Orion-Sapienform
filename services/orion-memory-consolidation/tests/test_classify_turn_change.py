import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    path = SERVICE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


classify_mod = _load("app/classify.py", "memory_classify")
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1


def _llm_raw(content: str, *, novel_lp=-0.2, shift_token="NONE") -> dict:
    novel_token = "NO"
    shift_parsed = "NONE"
    for line in content.splitlines():
        if line.startswith("NOVEL:"):
            novel_token = line.split(":", 1)[1].strip().upper()
        elif line.startswith("SHIFT:"):
            shift_parsed = line.split(":", 1)[1].strip().upper()
    shift_token = shift_parsed or shift_token
    shift_tops = [
        {"token": shift_token, "logprob": -0.3},
        {"token": "NONE", "logprob": -2.0},
        {"token": "STANCE", "logprob": -2.5},
        {"token": "REPAIR", "logprob": -3.0},
    ]
    if shift_token != "TOPIC":
        shift_tops.insert(2, {"token": "TOPIC", "logprob": -2.5})
    yes_lp = novel_lp if novel_token == "YES" else -2.0
    no_lp = -0.2 if novel_token == "NO" else -2.0
    return {
        "content": content,
        "raw": {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "NOVEL:", "logprob": -0.1},
                            {
                                "token": novel_token,
                                "logprob": novel_lp,
                                "top_logprobs": [
                                    {"token": "YES", "logprob": yes_lp},
                                    {"token": "NO", "logprob": no_lp},
                                ],
                            },
                            {"token": "SHIFT:", "logprob": -0.1},
                            {
                                "token": shift_token,
                                "logprob": -0.3,
                                "top_logprobs": shift_tops,
                            },
                            {"token": "MEMORY:", "logprob": -0.1},
                            {
                                "token": "NO",
                                "logprob": -0.3,
                                "top_logprobs": [
                                    {"token": "NO", "logprob": -0.3},
                                    {"token": "YES", "logprob": -2.0},
                                ],
                            },
                            {"token": "BOUNDARY:", "logprob": -0.1},
                            {
                                "token": "NO",
                                "logprob": -0.3,
                                "top_logprobs": [
                                    {"token": "NO", "logprob": -0.3},
                                    {"token": "YES", "logprob": -2.0},
                                ],
                            },
                        ]
                    }
                }
            ]
        },
    }


def test_settings_default_classify_route_is_metacog():
    from app.settings import Settings

    s = Settings()
    assert s.TURN_CHANGE_CLASSIFY_ROUTE == "metacog"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("metacog", "metacog"),
        ("METACOG", "metacog"),
        ("quick", "quick"),
        (" chat ", "metacog"),  # invalid → fallback
        ("chat-thinking", "metacog"),
        ("", "metacog"),
    ],
)
def test_resolve_classify_route_allowlist(raw, expected):
    from app.settings import Settings

    settings = Settings(TURN_CHANGE_CLASSIFY_ROUTE=raw)
    assert classify_mod._resolve_classify_route(settings) == expected


@pytest.mark.asyncio
async def test_classify_turn_rpc_uses_metacog_route_and_disables_thinking():
    bus = AsyncMock()
    content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    captured: dict = {}

    async def _rpc_request(channel, env, **kwargs):
        captured["channel"] = channel
        captured["env"] = env
        captured["kwargs"] = kwargs
        return {"data": b"x"}

    bus.rpc_request = _rpc_request

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, novel_lp=-2.0, shift_token="NONE")})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="more cats", response="still cute", spark_meta={}
    )
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    payload = captured["env"].payload
    assert payload["route"] == "metacog"
    assert payload["options"]["llm_route"] == "metacog"
    assert payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["options"]["return_logprobs"] is True
    assert patch["turn_change_classify_route"] == "metacog"


@pytest.mark.asyncio
async def test_classify_turn_invalid_route_falls_back_to_metacog(monkeypatch):
    bus = AsyncMock()
    content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    captured: dict = {}

    async def _rpc_request(channel, env, **kwargs):
        captured["env"] = env
        return {"data": b"x"}

    bus.rpc_request = _rpc_request

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, novel_lp=-2.0, shift_token="NONE")})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="more cats", response="still cute", spark_meta={}
    )
    from app.settings import Settings

    bad_settings = Settings(TURN_CHANGE_CLASSIFY_ROUTE="chat-thinking")
    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=bad_settings)
    payload = captured["env"].payload
    assert payload["route"] == "metacog"
    assert payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert patch["turn_change_classify_route"] == "metacog"


@pytest.mark.asyncio
async def test_classify_turn_quick_route_override():
    bus = AsyncMock()
    content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    captured: dict = {}

    async def _rpc_request(channel, env, **kwargs):
        captured["env"] = env
        return {"data": b"x"}

    bus.rpc_request = _rpc_request

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, novel_lp=-2.0, shift_token="NONE")})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="more cats", response="still cute", spark_meta={}
    )
    from app.settings import Settings

    quick_settings = Settings(TURN_CHANGE_CLASSIFY_ROUTE="quick")
    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=quick_settings)
    payload = captured["env"].payload
    assert payload["route"] == "quick"
    assert payload["options"]["llm_route"] == "quick"
    assert payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert patch["turn_change_classify_route"] == "quick"


@pytest.mark.asyncio
async def test_classify_turn_first_turn_baseline_none():
    bus = AsyncMock()
    bus.codec.decode.return_value.ok = True
    turn = MemoryTurnPersistedV1(correlation_id=str(uuid4()), prompt="hi", response="hello", spark_meta={})
    settings = classify_mod.settings if hasattr(classify_mod, "settings") else None
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=[], settings=app_settings)
    assert "turn_change_appraisal" in patch
    assert patch["turn_change_appraisal"]["baseline_mode"] == "none"
    assert patch["turn_change_appraisal"]["turn_change_status"] == "skipped"
    assert patch["turn_change_appraisal"]["novelty_score"] is None
    assert patch["turn_change_classify_route"] == "metacog"
    bus.rpc_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_classify_turn_routine_followup_low_novelty():
    bus = AsyncMock()
    content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    bus.rpc_request.return_value = {"data": b"x"}

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, novel_lp=-2.0, shift_token="NONE")})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="more cats", response="still cute", spark_meta={}
    )
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    appr = patch["turn_change_appraisal"]
    assert appr["turn_change_status"] == "ok"
    assert appr["baseline_mode"] in {"prior_turn", "session_window"}
    assert appr["novelty_score"] < 0.5


@pytest.mark.asyncio
async def test_classify_turn_topic_pivot_high_novelty():
    bus = AsyncMock()
    content = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: YES\nBOUNDARY: NO\n"
    bus.rpc_request.return_value = {"data": b"x"}

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, shift_token="TOPIC")})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="let's talk kubernetes", response="ok", spark_meta={}
    )
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    appr = patch["turn_change_appraisal"]
    assert appr["shift_kind"] == "TOPIC"
    assert appr["novelty_score"] >= 0.65


@pytest.mark.asyncio
async def test_classify_turn_text_fallback_marks_degraded():
    bus = AsyncMock()
    content = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: NO\nBOUNDARY: NO\n"
    bus.rpc_request.return_value = {"data": b"x"}

    def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": {"content": content, "raw": {"choices": [{"logprobs": {"content": []}}]}}})()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(correlation_id=str(uuid4()), prompt="pivot", response="ok", spark_meta={})
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    appr = patch["turn_change_appraisal"]
    assert appr["turn_change_status"] == "degraded"
    assert appr["novelty_score"] == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_classify_turn_low_margin_triggers_session_window_reappraisal():
    bus = AsyncMock()
    first_content = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: NO\nBOUNDARY: NO\n"
    retry_content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    call_count = 0
    captured_envs: list = []

    async def _rpc_request(channel, env, **kwargs):
        captured_envs.append(env)
        return {"data": b"x"}

    bus.rpc_request = _rpc_request

    def _decode_side_effect(_):
        nonlocal call_count
        call_count += 1
        content = first_content if call_count == 1 else retry_content
        class _R:
            ok = True
            envelope = type(
                "E",
                (),
                {
                    "payload": _llm_raw(
                        content,
                        novel_lp=-2.0 if call_count == 1 else -0.2,
                        shift_token="TOPIC" if call_count == 1 else "NONE",
                    )
                },
            )()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [
        {"correlation_id": "prev1", "prompt": "cats", "response": "cute"},
        {"correlation_id": "prev2", "prompt": "dogs", "response": "fun"},
    ]
    turn = MemoryTurnPersistedV1(
        correlation_id=str(uuid4()), prompt="maybe pivot", response="ok", spark_meta={}
    )
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    appr = patch["turn_change_appraisal"]
    assert len(captured_envs) == 2
    for env in captured_envs:
        assert env.payload["route"] == "metacog"
        assert env.payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert appr["baseline_mode"] == "session_window"
    assert appr["novelty_score"] < 0.5


@pytest.mark.asyncio
async def test_classify_turn_llm_failure_preserves_baseline_context():
    bus = AsyncMock()
    bus.rpc_request.side_effect = RuntimeError("llm down")
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(correlation_id=str(uuid4()), prompt="hi", response="hello", spark_meta={})
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    appr = patch["turn_change_appraisal"]
    assert appr["turn_change_status"] == "degraded"
    assert appr["baseline_mode"] == "prior_turn"
    assert appr["prior_correlation_id"] == "prev"
    assert patch["turn_change_classify_route"] == "metacog"


@pytest.mark.asyncio
async def test_worker_spark_meta_patch_includes_classify_route(monkeypatch):
    worker = _load("app/worker.py", "memory_consolidation_worker")
    published = []

    bus = AsyncMock()

    async def _publish(channel, env):
        published.append((channel, env))

    bus.publish = _publish

    async def _fake_classify(bus, *, turn, prior_turns, settings):
        return {
            "turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.2},
            "turn_change_classify_route": "metacog",
            "memory_classify_status": "ok",
        }

    monkeypatch.setattr(worker, "classify_turn", _fake_classify)

    window_store = AsyncMock()
    window_store._get_open_window = AsyncMock(return_value=None)
    window_store.get_window_turns = AsyncMock(return_value=[])
    window_store.append_turn = AsyncMock()
    suggest_runner = AsyncMock()

    corr = str(uuid4())
    turn = MemoryTurnPersistedV1(correlation_id=corr, prompt="hi", response="hello", spark_meta={})
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    env = BaseEnvelope(
        kind="memory.turn.persisted.v1",
        correlation_id=corr,
        source=ServiceRef(name="sql-writer", version="0.1", node="local"),
        payload=turn.model_dump(mode="json"),
    )

    await worker.handle_memory_turn_persisted(
        env, bus=bus, window_store=window_store, suggest_runner=suggest_runner
    )

    patch_envs = [e for ch, e in published if "spark_meta:patch" in ch]
    assert len(patch_envs) == 1
    assert patch_envs[0].payload["spark_meta"]["turn_change_classify_route"] == "metacog"


@pytest.mark.asyncio
async def test_worker_skips_substrate_emit_low_confidence(monkeypatch):
    worker = _load("app/worker.py", "memory_consolidation_worker")
    published = []

    bus = AsyncMock()

    async def _publish(channel, env):
        published.append((channel, env))

    bus.publish = _publish

    appraisal = {
        "turn_change_status": "ok",
        "novelty_score": 0.9,
        "shift_kind": "TOPIC",
        "confidence": 0.04,
    }

    async def _fake_classify(bus, *, turn, prior_turns, settings):
        return {"turn_change_appraisal": appraisal, "memory_classify_status": "ok"}

    monkeypatch.setattr(worker, "classify_turn", _fake_classify)

    window_store = AsyncMock()
    window_store._get_open_window = AsyncMock(return_value=None)
    window_store.get_window_turns = AsyncMock(return_value=[])
    window_store.append_turn = AsyncMock()
    suggest_runner = AsyncMock()

    corr = str(uuid4())
    turn = MemoryTurnPersistedV1(correlation_id=corr, prompt="hi", response="hello", spark_meta={})
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    env = BaseEnvelope(
        kind="memory.turn.persisted.v1",
        correlation_id=corr,
        source=ServiceRef(name="sql-writer", version="0.1", node="local"),
        payload=turn.model_dump(mode="json"),
    )

    await worker.handle_memory_turn_persisted(
        env, bus=bus, window_store=window_store, suggest_runner=suggest_runner
    )

    signal_kinds = [e.kind for _, e in published]
    assert "signal.memory_consolidation.turn_change" not in signal_kinds


def test_session_window_baseline_respects_turn_cap():
    prior = [
        {"correlation_id": "t1", "prompt": "one", "response": "r1"},
        {"correlation_id": "t2", "prompt": "two", "response": "r2"},
        {"correlation_id": "t3", "prompt": "three", "response": "r3"},
        {"correlation_id": "t4", "prompt": "four", "response": "r4"},
    ]
    _, text = classify_mod._session_window_baseline(prior, n=2)
    assert "three" in text
    assert "four" in text
    assert "one" not in text
    assert "two" not in text


def test_prior_turn_baseline_clips_long_fields():
    long_prompt = "x" * 1000
    prior = [{"correlation_id": "prev", "prompt": long_prompt, "response": "ok"}]
    _, text, corr = classify_mod._prior_turn_baseline(prior)
    assert corr == "prev"
    assert len(text) < len(long_prompt)
    assert "..." in text


@pytest.mark.asyncio
async def test_classify_turn_reappraisal_failure_keeps_primary_scores():
    bus = AsyncMock()
    first_content = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: NO\nBOUNDARY: NO\n"
    call_count = 0
    bus.rpc_request.return_value = {"data": b"x"}

    def _decode_side_effect(_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            payload = _llm_raw(first_content, novel_lp=-2.0, shift_token="TOPIC")
            class _R:
                ok = True
                envelope = type("E", (), {"payload": payload})()
            return _R()
        raise RuntimeError("reappraisal down")

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [
        {"correlation_id": "prev1", "prompt": "cats", "response": "cute"},
        {"correlation_id": "prev2", "prompt": "dogs", "response": "fun"},
    ]
    turn = MemoryTurnPersistedV1(correlation_id=str(uuid4()), prompt="pivot", response="ok", spark_meta={})
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    appr = patch["turn_change_appraisal"]
    assert bus.rpc_request.await_count == 2
    assert appr["baseline_mode"] == "prior_turn"
    assert appr["novelty_score"] == pytest.approx(0.5, abs=0.05)
    assert "turn_change_appraisal" in patch


@pytest.mark.asyncio
async def test_classify_turn_low_margin_single_prior_triggers_reappraisal():
    bus = AsyncMock()
    first_content = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: NO\nBOUNDARY: NO\n"
    retry_content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    call_count = 0
    bus.rpc_request.return_value = {"data": b"x"}

    def _decode_side_effect(_):
        nonlocal call_count
        call_count += 1
        content = first_content if call_count == 1 else retry_content
        class _R:
            ok = True
            envelope = type(
                "E",
                (),
                {
                    "payload": _llm_raw(
                        content,
                        novel_lp=-2.0 if call_count == 1 else -0.2,
                        shift_token="TOPIC" if call_count == 1 else "NONE",
                    )
                },
            )()

        return _R()

    bus.codec.decode = Mock(side_effect=_decode_side_effect)
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(correlation_id=str(uuid4()), prompt="pivot", response="ok", spark_meta={})
    from app.settings import settings as app_settings

    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=app_settings)
    assert bus.rpc_request.await_count == 2
    assert patch["turn_change_appraisal"]["baseline_mode"] == "session_window"


@pytest.mark.asyncio
async def test_worker_emits_substrate_signal_above_threshold(monkeypatch):
    worker = _load("app/worker.py", "memory_consolidation_worker")
    published = []

    bus = AsyncMock()

    async def _publish(channel, env):
        published.append((channel, env))

    bus.publish = _publish

    appraisal = {
        "turn_change_status": "ok",
        "novelty_score": 0.9,
        "shift_kind": "TOPIC",
        "confidence": 0.88,
    }

    async def _fake_classify(bus, *, turn, prior_turns, settings):
        return {
            "turn_change_appraisal": appraisal,
            "memory_significance_score": 0.5,
            "conversation_boundary_score": 0.1,
            "memory_classify_status": "ok",
        }

    monkeypatch.setattr(worker, "classify_turn", _fake_classify)

    window_store = AsyncMock()
    window_store._get_open_window = AsyncMock(return_value={"memory_window_id": "w1"})
    window_store.get_window_turns = AsyncMock(return_value=[])
    window_store.append_turn = AsyncMock()
    window_store.close_current_window = AsyncMock()
    suggest_runner = AsyncMock()

    corr = str(uuid4())
    turn = MemoryTurnPersistedV1(correlation_id=corr, prompt="hi", response="hello", spark_meta={})
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    env = BaseEnvelope(
        kind="memory.turn.persisted.v1",
        correlation_id=corr,
        source=ServiceRef(name="sql-writer", version="0.1", node="local"),
        payload=turn.model_dump(mode="json"),
    )

    await worker.handle_memory_turn_persisted(
        env,
        bus=bus,
        window_store=window_store,
        suggest_runner=suggest_runner,
    )

    signal_kinds = [env.kind for _, env in published]
    assert "signal.memory_consolidation.turn_change" in signal_kinds


@pytest.mark.asyncio
async def test_worker_append_turn_when_substrate_publish_fails(monkeypatch):
    worker = _load("app/worker.py", "memory_consolidation_worker")

    bus = AsyncMock()
    publish_calls = []

    async def _publish(channel, env):
        publish_calls.append((channel, env))
        if "signals" in channel:
            raise RuntimeError("bus down")

    bus.publish = _publish

    appraisal = {
        "turn_change_status": "ok",
        "novelty_score": 0.9,
        "shift_kind": "TOPIC",
        "confidence": 0.88,
    }

    async def _fake_classify(bus, *, turn, prior_turns, settings):
        return {"turn_change_appraisal": appraisal, "memory_classify_status": "ok"}

    monkeypatch.setattr(worker, "classify_turn", _fake_classify)

    window_store = AsyncMock()
    window_store._get_open_window = AsyncMock(return_value=None)
    window_store.get_window_turns = AsyncMock(return_value=[])
    window_store.append_turn = AsyncMock()
    suggest_runner = AsyncMock()

    corr = str(uuid4())
    turn = MemoryTurnPersistedV1(correlation_id=corr, prompt="hi", response="hello", spark_meta={})
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    env = BaseEnvelope(
        kind="memory.turn.persisted.v1",
        correlation_id=corr,
        source=ServiceRef(name="sql-writer", version="0.1", node="local"),
        payload=turn.model_dump(mode="json"),
    )

    await worker.handle_memory_turn_persisted(
        env, bus=bus, window_store=window_store, suggest_runner=suggest_runner
    )

    window_store.append_turn.assert_awaited_once()
