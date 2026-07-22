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

    async def connect(self) -> None:
        self.connected = True

    async def rpc_request(self, *args, **kwargs):
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
    fetch_id_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(mod, "_get_memory_pool", fake_pool)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)
    monkeypatch.setattr(mod.mc_dal, "fetch_card_id_by_fingerprint", fetch_id_mock)

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
    fetch_id_mock = AsyncMock(return_value="44444444-4444-4444-4444-444444444444")
    reconfirm_mock = AsyncMock()
    monkeypatch.setattr(mod, "_get_memory_pool", fake_pool)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)
    monkeypatch.setattr(mod.mc_dal, "fetch_card_id_by_fingerprint", fetch_id_mock)
    monkeypatch.setattr(mod.mc_dal, "record_reconfirmation", reconfirm_mock)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_not_called()
    reconfirm_mock.assert_awaited_once()


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


def _wire_common(monkeypatch, mod, *, insert_mock=None, exists_mock=None, fetch_id_mock=None, reconfirm_mock=None):
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
    # Default: no existing card for this fingerprint (None), matching
    # exists_mock's default False -- tests that care about the dedup/
    # reconfirmation path override this explicitly.
    fetch_id_mock = fetch_id_mock or AsyncMock(return_value=None)
    reconfirm_mock = reconfirm_mock or AsyncMock()
    monkeypatch.setattr(mod.mc_dal, "fetch_card_id_by_fingerprint", fetch_id_mock)
    monkeypatch.setattr(mod.mc_dal, "record_reconfirmation", reconfirm_mock)
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
    assert isinstance(resolved, Path)


def test_dedup_hit_records_reconfirmation_instead_of_silent_skip(monkeypatch) -> None:
    """Stage 2 phase 1 (2026-07-22): a fingerprint dedup-hit on the LLM
    annotation path must record a reconfirmation (with the turn's real
    session_id) instead of the old silent skip that discarded this
    information entirely -- and must NOT create a duplicate card."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)
    existing_card_id = "11111111-1111-1111-1111-111111111111"
    fetch_id_mock = AsyncMock(return_value=existing_card_id)
    reconfirm_mock = AsyncMock()
    monkeypatch.setattr(mod.mc_dal, "fetch_card_id_by_fingerprint", fetch_id_mock)
    monkeypatch.setattr(mod.mc_dal, "record_reconfirmation", reconfirm_mock)

    fake_bus = _FakeBus(json.dumps(_VALID_ANNOTATION))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    env = _turn_env(prompt="I live in Ogden, UT")
    env.payload["session_id"] = "session-xyz"
    asyncio.run(mod.handle_memory_history_turn(env))

    insert_mock.assert_not_called()  # no duplicate card
    reconfirm_mock.assert_awaited_once()
    assert reconfirm_mock.await_args.kwargs["card_id"] == existing_card_id
    assert reconfirm_mock.await_args.kwargs["session_id"] == "session-xyz"
    assert reconfirm_mock.await_args.kwargs["actor"] == "auto_extractor"


def test_dedup_hit_on_regex_fallback_path_also_records_reconfirmation(monkeypatch) -> None:
    """Same guarantee on the regex-fallback path (LLM unavailable), not
    just the LLM-annotation path -- both dedup sites must be covered."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)
    existing_card_id = "22222222-2222-2222-2222-222222222222"
    fetch_id_mock = AsyncMock(return_value=existing_card_id)
    reconfirm_mock = AsyncMock()
    monkeypatch.setattr(mod.mc_dal, "fetch_card_id_by_fingerprint", fetch_id_mock)
    monkeypatch.setattr(mod.mc_dal, "record_reconfirmation", reconfirm_mock)

    fake_bus = _FakeBus("", raise_on_rpc=RuntimeError("bus unavailable"))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    env = _turn_env(prompt="I live in Ogden, UT")
    env.payload["session_id"] = "session-regex"
    asyncio.run(mod.handle_memory_history_turn(env))

    insert_mock.assert_not_called()
    reconfirm_mock.assert_awaited_once()
    assert reconfirm_mock.await_args.kwargs["session_id"] == "session-regex"


def test_reconfirmation_record_failure_does_not_crash_turn_handling(monkeypatch) -> None:
    """Matches every other write in this module's fail-open contract -- a
    DB error recording the reconfirmation must not propagate and abort
    turn processing."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)
    monkeypatch.setattr(
        mod.mc_dal, "fetch_card_id_by_fingerprint", AsyncMock(return_value="33333333-3333-3333-3333-333333333333")
    )
    monkeypatch.setattr(mod.mc_dal, "record_reconfirmation", AsyncMock(side_effect=RuntimeError("db down")))

    fake_bus = _FakeBus(json.dumps(_VALID_ANNOTATION))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    # Must not raise.
    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))
    insert_mock.assert_not_called()


def test_new_card_creation_stashes_session_id_in_subschema(monkeypatch) -> None:
    """The card's own creation session must be recorded too (subschema.
    auto_extractor_session_id), not just later reconfirmations -- otherwise
    count_distinct_reconfirmation_sessions can never count a card's
    original session as one of the distinct sessions confirming it."""
    import app.memory_extractor as mod

    insert_mock, _ = _wire_common(monkeypatch, mod)
    fake_bus = _FakeBus(json.dumps(_VALID_ANNOTATION))
    monkeypatch.setattr(mod, "_get_annotation_bus", lambda: fake_bus)

    env = _turn_env(prompt="I live in Ogden, UT")
    env.payload["session_id"] = "session-creation"
    asyncio.run(mod.handle_memory_history_turn(env))

    insert_mock.assert_awaited_once()
    card_arg = insert_mock.await_args.args[1]
    assert (card_arg.subschema or {}).get("auto_extractor_session_id") == "session-creation"
