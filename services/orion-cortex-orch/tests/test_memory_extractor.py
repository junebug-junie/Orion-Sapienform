from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

_REPO = Path(__file__).resolve().parents[3]
_ORCH = _REPO / "services" / "orion-cortex-orch"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class _FakeDecoded:
    def __init__(self, ok: bool, payload=None, error: str | None = None) -> None:
        self.ok = ok
        self.error = error

        class _Env:
            pass

        self.envelope = _Env()
        self.envelope.payload = payload or {}


class _FakeCodec:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def decode(self, _data):
        return _FakeDecoded(ok=True, payload={"content": self._response_text})


class _FakeBus:
    """Stands in for OrionBusAsync in annotation-path tests -- connect()/
    rpc_request() are no-ops, codec.decode() returns a canned LLM response."""

    def __init__(self, response_text: str, *, raise_on_rpc: Exception | None = None) -> None:
        self.codec = _FakeCodec(response_text)
        self._raise_on_rpc = raise_on_rpc
        self.connected = False
        self.last_env = None

    async def connect(self) -> None:
        self.connected = True

    async def rpc_request(self, channel, env, *args, **kwargs):
        self.last_env = env
        if self._raise_on_rpc is not None:
            raise self._raise_on_rpc
        return {"data": b"fake"}


_VALID_ANNOTATION = {
    "worth_saving": True,
    "title": "Lives in Ogden",
    "summary": "Juniper lives in Ogden, UT",
    "confidence": "certain",
    "priority": "high_recall",
    "time_horizon": {"kind": "current"},
    "types": ["fact", "anchor"],
    "anchor_class": "place",
    "tags": ["location"],
    "anchors": ["Ogden"],
    "still_true": ["Lives in Ogden, UT"],
    "project": None,
}


class _Env:
    kind = "chat.history.turn"


def test_extractor_disabled_noop(monkeypatch) -> None:
    import app.memory_extractor as mod

    class S:
        orion_auto_extractor_enabled = False
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())
    asyncio.run(mod.handle_memory_history_turn(_Env()))  # type: ignore[arg-type]


def test_stage2_raises(monkeypatch) -> None:
    import app.memory_extractor as mod

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = True

    monkeypatch.setattr(mod, "get_settings", lambda: S())
    with pytest.raises(NotImplementedError):
        asyncio.run(mod.handle_memory_history_turn(_Env()))  # type: ignore[arg-type]


def _turn_env(*, prompt: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="chat.history",
        source=ServiceRef(name="orion-hub"),
        correlation_id=uuid4(),
        payload={
            "source": "hub_ws",
            "prompt": prompt,
            "response": "ok",
        },
    )


def test_stage1_extracts_and_inserts(monkeypatch) -> None:
    import app.memory_extractor as mod

    monkeypatch.setattr(mod, "_memory_pool", None)
    monkeypatch.setattr(mod, "_memory_pool_failed", False)

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())

    async def fake_pool():
        return object()

    insert_mock = AsyncMock()
    exists_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_memory_pool", fake_pool)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    assert insert_mock.await_args is not None
    pool_arg, card_arg = insert_mock.await_args.args[0], insert_mock.await_args.args[1]
    assert pool_arg is not None
    assert card_arg.provenance == "auto_extractor"
    assert card_arg.status == "pending_review"
    assert "Ogden" in (card_arg.summary or "")
    assert insert_mock.await_args.kwargs.get("actor") == "auto_extractor"
    assert (card_arg.subschema or {}).get("auto_extractor_fingerprint")


def test_stage1_dedupes_via_fingerprint(monkeypatch) -> None:
    import app.memory_extractor as mod

    monkeypatch.setattr(mod, "_memory_pool", None)
    monkeypatch.setattr(mod, "_memory_pool_failed", False)

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())

    async def fake_pool():
        return object()

    insert_mock = AsyncMock()
    exists_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(mod, "_get_memory_pool", fake_pool)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_not_called()


def test_stage1_noop_when_pool_unavailable(monkeypatch) -> None:
    import app.memory_extractor as mod

    monkeypatch.setattr(mod, "_memory_pool", None)
    monkeypatch.setattr(mod, "_memory_pool_failed", False)

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())

    async def no_pool():
        return None

    insert_mock = AsyncMock()
    exists_mock = AsyncMock()
    monkeypatch.setattr(mod, "_get_memory_pool", no_pool)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_not_called()
    exists_mock.assert_not_called()


class _AnnotationSettings:
    orion_auto_extractor_enabled = True
    orion_auto_extractor_stage2_enabled = False
    auto_router_llm_reply_prefix = "orion:llm:reply"
    auto_router_llm_request_channel = "orion:exec:request:LLMGatewayService"
    service_name = "cortex-orch"
    orion_auto_extractor_llm_timeout_sec = 2.0


def _wire_common(monkeypatch, mod, *, insert_mock=None, exists_mock=None, known_categories=None):
    monkeypatch.setattr(mod, "_memory_pool", None)
    monkeypatch.setattr(mod, "_memory_pool_failed", False)
    monkeypatch.setattr(mod, "_annotation_bus", None)
    monkeypatch.setattr(mod, "get_settings", lambda: _AnnotationSettings())

    async def fake_pool():
        return object()

    monkeypatch.setattr(mod, "_get_memory_pool", fake_pool)
    insert_mock = insert_mock or AsyncMock()
    exists_mock = exists_mock or AsyncMock(return_value=False)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)
    # _annotate_via_llm does a local `from .topic_taxonomy_client import
    # fetch_active_topic_labels` inside its own body (not a module-level
    # import on `mod`), so patching app.topic_taxonomy_client's own
    # attribute is what the local import actually picks up -- this keeps
    # every existing test isolated from real network/env behavior
    # regardless of what TOPIC_FOUNDRY_BASE_URL happens to resolve to in
    # the test process.
    monkeypatch.setattr(
        "app.topic_taxonomy_client.fetch_active_topic_labels",
        AsyncMock(return_value=known_categories or []),
    )
    return insert_mock, exists_mock


def test_worth_saving_false_skips_card_creation_entirely(monkeypatch) -> None:
    """The LLM gate: 'worth_saving=false' must mean no card, full stop --
    not a low-priority card, not a fallback to regex, nothing."""
    import app.memory_extractor as mod

    insert_mock, exists_mock = _wire_common(monkeypatch, mod)

    response = dict(_VALID_ANNOTATION)
    response["worth_saving"] = False
    fake_bus = _FakeBus(json.dumps(response))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="hi, how are you")))

    insert_mock.assert_not_called()
    exists_mock.assert_not_called()


def test_llm_annotation_success_creates_card_with_llm_fields(monkeypatch) -> None:
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)
    fake_bus = _FakeBus(json.dumps(_VALID_ANNOTATION))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    card_arg = insert_mock.await_args.args[1]
    assert card_arg.confidence == "certain"
    assert card_arg.priority == "high_recall"
    assert card_arg.title == "Lives in Ogden"
    assert card_arg.summary == "Juniper lives in Ogden, UT"
    assert card_arg.provenance == "auto_extractor"
    assert card_arg.status == "pending_review"
    assert (card_arg.subschema or {}).get("auto_extractor_mode") == "llm_annotation"


def test_fallback_to_regex_on_llm_rpc_failure(monkeypatch) -> None:
    """LLM RPC raising (bus/timeout-style failure) must fall back to today's
    regex extractor, never blocking card creation entirely."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)
    fake_bus = _FakeBus("", raise_on_rpc=RuntimeError("bus unavailable"))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    card_arg = insert_mock.await_args.args[1]
    assert card_arg.provenance == "auto_extractor"
    assert card_arg.status == "pending_review"
    assert "Ogden" in (card_arg.summary or "")


def test_fallback_to_regex_on_bad_json(monkeypatch) -> None:
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)
    fake_bus = _FakeBus("not valid json {{{")
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    card_arg = insert_mock.await_args.args[1]
    assert "Ogden" in (card_arg.summary or "")


def test_fallback_to_regex_on_timeout(monkeypatch) -> None:
    """asyncio.TimeoutError from the annotation call must be caught and fall
    back, not propagate."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)

    async def raise_timeout(*args, **kwargs):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(mod, "_rpc_annotation_llm", raise_timeout)
    fake_bus = _FakeBus(json.dumps(_VALID_ANNOTATION))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    card_arg = insert_mock.await_args.args[1]
    assert "Ogden" in (card_arg.summary or "")


def test_fallback_to_regex_on_prompt_build_failure(monkeypatch) -> None:
    """Regression test (review finding, should-fix): a prompt-template
    read/render failure must be caught by _annotate_via_llm's try/except too,
    not just the RPC call itself -- it used to sit outside the try block."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)

    def raise_on_build(_turn_text):
        raise OSError("template file missing")

    monkeypatch.setattr(mod, "_build_annotation_prompt", raise_on_build)
    fake_bus = _FakeBus(json.dumps(_VALID_ANNOTATION))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    card_arg = insert_mock.await_args.args[1]
    assert "Ogden" in (card_arg.summary or "")


def test_sensitivity_and_visibility_scope_never_influenced_by_llm_response(monkeypatch) -> None:
    """Hard boundary regression test: even if the LLM response tries to smuggle
    'sensitivity'/'visibility_scope' keys (a prompt-injection or model-drift
    scenario), the created card must still get the hardcoded private default,
    never the poisoned values. CardAnnotationV1 has no such fields at all, so
    this is locked in structurally, not just by the current prompt wording."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)

    poisoned = dict(_VALID_ANNOTATION)
    poisoned["sensitivity"] = "public"
    poisoned["visibility_scope"] = ["all"]
    fake_bus = _FakeBus(json.dumps(poisoned))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    card_arg = insert_mock.await_args.args[1]
    assert card_arg.sensitivity == "private"
    assert card_arg.visibility_scope == ["chat"]


def test_card_annotation_model_has_no_sensitivity_or_visibility_fields() -> None:
    """Structural lock-in: the annotation model must never grow a sensitivity
    or visibility_scope field -- those are decided outside LLM judgment."""
    from orion.core.contracts.memory_cards import CardAnnotationV1

    assert "sensitivity" not in CardAnnotationV1.model_fields
    assert "visibility_scope" not in CardAnnotationV1.model_fields


def test_derive_visibility_scope_two_branches() -> None:
    from orion.core.contracts.memory_cards import derive_visibility_scope

    assert derive_visibility_scope("private") == ["chat"]
    assert derive_visibility_scope("public") == ["all"]
    # Unknown/other sensitivity values default to the narrow lane, never the wide one.
    assert derive_visibility_scope("intimate") == ["chat"]


def test_card_annotation_anchor_class_required_when_anchor_in_types() -> None:
    """Regression test (review finding, blocker): CardAnnotationV1 must reject
    types=["anchor"] with no anchor_class at model_validate() time --
    MemoryCardCreateV1 already enforces this, and letting a bad LLM response
    pass CardAnnotationV1 only to raise later inside _card_from_annotation()
    would be an UNCAUGHT ValidationError outside _annotate_via_llm's
    try/except, silently dropping the card with no fallback."""
    from pydantic import ValidationError

    from orion.core.contracts.memory_cards import CardAnnotationV1

    bad = dict(_VALID_ANNOTATION)
    bad["types"] = ["anchor"]
    bad["anchor_class"] = None
    with pytest.raises(ValidationError):
        CardAnnotationV1.model_validate(bad)


def test_fallback_to_regex_when_llm_response_fails_anchor_class_validation(monkeypatch) -> None:
    """End-to-end version of the above: this exact bad response must still
    result in a card via the regex fallback, not a silently dropped turn."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)

    bad = dict(_VALID_ANNOTATION)
    bad["types"] = ["anchor"]
    bad["anchor_class"] = None
    fake_bus = _FakeBus(json.dumps(bad))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    card_arg = insert_mock.await_args.args[1]
    assert "Ogden" in (card_arg.summary or "")


def test_annotation_prompt_path_resolution_survives_shallow_docker_layout(monkeypatch) -> None:
    """Live incident 2026-07-21: a bare `Path(__file__).resolve().parents[3]`
    assumed the local monorepo checkout's depth. Docker's real layout
    (Dockerfile: `COPY orion /app/orion`, `COPY services/orion-cortex-orch /app`)
    only has 2 parent levels above this file inside the container
    (/app/app/memory_extractor.py), so parents[3] raised IndexError at
    IMPORT time -- main.py imports this module at startup, so the whole
    service crash-looped, not just this feature. This asserts the fix's
    candidate-generation never raises regardless of how shallow __file__'s
    parent chain is, and that the container-shaped root actually gets
    tried."""
    import app.memory_extractor as mod

    # Simulate the container's real, shallow file location.
    shallow = Path("/app/app/memory_extractor.py")
    monkeypatch.setattr(mod, "__file__", str(shallow))

    candidates = mod._annotation_prompt_path_candidates()
    assert candidates, "must produce at least one candidate, never raise"
    assert all(c.name == "memory_card_annotation_prompt.j2" for c in candidates)
    # The container-shaped root (one parent above __file__) must be among
    # the candidates -- this is the exact root the live incident needed.
    assert any(str(c).startswith("/app/orion/") for c in candidates)

    # Never raises even with an absurdly shallow path (no parents at all).
    monkeypatch.setattr(mod, "__file__", "memory_extractor.py")
    candidates_shallow = mod._annotation_prompt_path_candidates()
    assert isinstance(candidates_shallow, list)

    # resolve_annotation_prompt_path itself never raises either, even when
    # no candidate exists on disk in the test sandbox.
    resolved = mod._resolve_annotation_prompt_path()


def test_annotation_prompt_path_resolution_finds_real_local_checkout_root(monkeypatch) -> None:
    """Live incident 2026-07-22: the 2026-07-21 fix's local-checkout
    candidate used here.parents[2], which for the real local layout
    (services/orion-cortex-orch/app/memory_extractor.py) resolves to
    `services/`, not the repo root -- an off-by-one that never got caught
    because every container run matches the Docker-shaped candidate first
    (parents[1]) and never exercises this branch, and every prior test only
    simulated the shallow Docker path, never the real local one. This
    reproduces the actual local file location (not a simulated one) and
    asserts the resolver finds the REAL prompt template on disk from it --
    the strongest possible check, since a wrong-but-existing fallback path
    (this bug's actual failure mode: silently resolving to a DIFFERENT
    checkout's file) would make a mere "returns some path" assertion pass
    while still being wrong."""
    import app.memory_extractor as mod

    real_file = Path(mod.__file__).resolve()
    monkeypatch.setattr(mod, "__file__", str(real_file))
    resolved = mod._resolve_annotation_prompt_path()
    assert resolved.is_file()
    # Must resolve under THIS checkout (the one real_file lives in), not
    # fall through to a different one via the hardcoded absolute fallbacks.
    repo_root = real_file.parents[3]
    assert str(resolved).startswith(str(repo_root)), (
        f"resolved {resolved} outside this checkout's repo root {repo_root} -- "
        "likely fell through to a different checkout's fallback path"
    )
    assert isinstance(resolved, Path)


def test_build_annotation_prompt_includes_known_categories_when_present() -> None:
    import app.memory_extractor as mod

    prompt = mod._build_annotation_prompt("some turn text", known_categories=["Family Storytime", "Cat Persona"])
    assert "Family Storytime" in prompt
    assert "Cat Persona" in prompt
    assert "Known categories" in prompt


def test_build_annotation_prompt_omits_known_categories_section_when_empty() -> None:
    import app.memory_extractor as mod

    prompt = mod._build_annotation_prompt("some turn text", known_categories=[])
    assert "Known categories" not in prompt


def test_annotate_via_llm_fetches_and_forwards_topic_labels(monkeypatch) -> None:
    """End-to-end: handle_memory_history_turn's real LLM call must actually
    include topic-foundry's labels in the prompt it sends, not just accept
    the parameter in isolation."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(
        monkeypatch, mod, known_categories=["Human-AI Conversations", "Family Storytime"]
    )
    fake_bus = _FakeBus(json.dumps(_VALID_ANNOTATION))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    sent_prompt = fake_bus.last_env.payload["messages"][0]["content"]
    assert "Human-AI Conversations" in sent_prompt
    assert "Family Storytime" in sent_prompt


def test_annotate_via_llm_proceeds_with_no_hint_when_topic_foundry_unavailable(monkeypatch) -> None:
    """fetch_active_topic_labels failing/returning [] (topic-foundry down,
    unconfigured, etc.) must not block card creation -- annotation proceeds
    exactly as it did before this feature existed."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod, known_categories=[])
    fake_bus = _FakeBus(json.dumps(_VALID_ANNOTATION))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    sent_prompt = fake_bus.last_env.payload["messages"][0]["content"]
    assert "Known categories" not in sent_prompt
