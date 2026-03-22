import asyncio
import os
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.verb_adapters import RespondToJuniperCollapseMirrorVerb  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.verbs.base import VerbContext  # noqa: E402
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2  # noqa: E402
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest  # noqa: E402
from orion.schemas.self_study import SelfStudyRetrieveResultV1  # noqa: E402


class _Codec:
    @staticmethod
    def decode(data):
        return SimpleNamespace(ok=True, error=None, envelope=BaseEnvelope.model_validate(data))


class _FakeBus:
    def __init__(self, recall_payload: dict, llm_payload: dict) -> None:
        self.codec = _Codec()
        self.recall_payload = recall_payload
        self.llm_payload = llm_payload
        self.calls = []

    async def rpc_request(self, channel: str, envelope: BaseEnvelope, *, reply_channel: str, timeout_sec: float):
        self.calls.append((channel, envelope))
        payload = self.recall_payload if "RecallService" in channel else self.llm_payload
        return {
            "data": BaseEnvelope(
                kind="result",
                source=ServiceRef(name="test"),
                correlation_id=str(envelope.correlation_id),
                payload=payload,
            ).model_dump(mode="json")
        }


@pytest.fixture
def collapse_request() -> PlanExecutionRequest:
    entry = CollapseMirrorEntryV2(
        observer="Juniper",
        trigger="something shifted",
        observer_state=["steady"],
        type="reflect",
        emergent_entity="signal",
        summary="Juniper noticed a pattern.",
        mantra="stay grounded",
        tags=["juniper", "mirror"],
    )
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="actions.respond_to_juniper_collapse_mirror.v1",
            steps=[],
        ),
        args=PlanExecutionArgs(request_id=str(uuid4()), extra={"mode": "brain"}),
        context={
            "metadata": {
                "collapse_entry": entry.model_dump(mode="json"),
                "recipient_group": "juniper_primary",
                "session_id": "collapse_mirror",
                "notify_dedupe_key": f"actions:collapse_reply:{entry.event_id}",
                "notify_dedupe_window_seconds": 86400,
                "recall_profile": "reflect.v1",
            }
        },
    )



def test_actions_verb_builds_notify_request_chat_message(monkeypatch, collapse_request):
    sent = {}

    def _send(self, request):
        sent["request"] = request
        return SimpleNamespace(ok=True, status="accepted", detail=None, notification_id="notif-1")

    monkeypatch.setattr("app.verb_adapters.NotifyClient.send", _send)
    bus = _FakeBus(
        recall_payload={"bundle": {"rendered": "memory block", "items": []}},
        llm_payload={"content": "[MESSAGE]\nhello Juniper\n[/MESSAGE]"},
    )
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": str(uuid4())})

    output, effects = asyncio.run(RespondToJuniperCollapseMirrorVerb().execute(ctx, collapse_request))

    assert effects == []
    assert output.status == "success"
    assert sent["request"].event_kind == "orion.chat.message"
    assert sent["request"].session_id == "collapse_mirror"
    assert sent["request"].recipient_group == "juniper_primary"
    assert sent["request"].dedupe_key.startswith("actions:collapse_reply:")



def test_actions_verb_prompt_contains_relevant_memory_marker(monkeypatch, collapse_request):
    monkeypatch.setattr(
        "app.verb_adapters.NotifyClient.send",
        lambda self, request: SimpleNamespace(ok=True, status="accepted", detail=None, notification_id="notif-1"),
    )
    bus = _FakeBus(
        recall_payload={"bundle": {"rendered": "memory block", "items": []}},
        llm_payload={"content": "[MESSAGE]\nhello Juniper\n[/MESSAGE]"},
    )
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": str(uuid4())})

    asyncio.run(RespondToJuniperCollapseMirrorVerb().execute(ctx, collapse_request))

    llm_channel, llm_envelope = bus.calls[-1]
    assert "LLMGatewayService" in llm_channel
    messages = llm_envelope.payload["messages"]
    assert "RELEVANT MEMORY" in messages[1]["content"]
    assert "memory block" in messages[1]["content"]


def test_actions_verb_reflective_self_study_is_explicit_and_preserves_metadata(monkeypatch, collapse_request):
    async def _fake_self_retrieve(*, request, **_kwargs):
        return SelfStudyRetrieveResultV1.model_validate(
            {
                "run_id": "run-1",
                "retrieval_mode": request.retrieval_mode,
                "applied_filters": request.filters.model_dump(),
                "groups": [
                    {
                        "trust_tier": "authoritative",
                        "items": [
                            {
                                "stable_id": "fact-1",
                                "trust_tier": "authoritative",
                                "record_type": "fact",
                                "title": "Mirror action entrypoint",
                                "content_preview": "The action verb sends a notification after LLM generation.",
                                "source_kind": "self_study",
                                "source_snapshot_id": "snapshot-1",
                                "source_path": "services/orion-cortex-exec/app/verb_adapters.py",
                                "evidence": [],
                                "concept_refs": [],
                                "metadata": {"provenance": ["repo"]},
                            }
                        ],
                    },
                    {
                        "trust_tier": "induced",
                        "items": [
                            {
                                "stable_id": "concept-1",
                                "trust_tier": "induced",
                                "record_type": "concept",
                                "title": "Mirror workflow concept",
                                "content_preview": "The action combines recall, reflective self-study, and messaging.",
                                "source_kind": "self_study",
                                "source_snapshot_id": "snapshot-1",
                                "source_path": "services/orion-cortex-exec/app/verb_adapters.py",
                                "evidence": [],
                                "concept_refs": [],
                                "metadata": {"provenance": ["repo"]},
                            }
                        ],
                    },
                    {
                        "trust_tier": "reflective",
                        "items": [
                            {
                                "stable_id": "reflection-1",
                                "trust_tier": "reflective",
                                "record_type": "reflection",
                                "title": "Reflective caution",
                                "content_preview": "Reflective material must stay labeled as reflective.",
                                "source_kind": "self_study",
                                "source_snapshot_id": "snapshot-1",
                                "source_path": "services/orion-cortex-exec/app/self_study.py",
                                "reflection_kind": "architecture_observation",
                                "evidence": [],
                                "concept_refs": [],
                                "metadata": {"provenance": ["repo"]},
                            }
                        ],
                    },
                ],
                "counts": {"total": 3, "authoritative": 1, "induced": 1, "reflective": 1, "facts": 1, "concepts": 1, "reflections": 1},
                "backend_status": [],
                "notes": [],
            }
        )

    monkeypatch.setattr(
        "app.verb_adapters.NotifyClient.send",
        lambda self, request: SimpleNamespace(ok=True, status="accepted", detail=None, notification_id="notif-1"),
    )
    monkeypatch.setattr("app.verb_adapters.run_self_retrieve", _fake_self_retrieve)

    collapse_request.context["metadata"]["self_study"] = {"enabled": True, "retrieval_mode": "reflective"}
    bus = _FakeBus(
        recall_payload={"bundle": {"rendered": "memory block", "items": []}},
        llm_payload={"content": "[MESSAGE]\nhello Juniper\n[/MESSAGE]"},
    )
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": str(uuid4())})

    output, _effects = asyncio.run(RespondToJuniperCollapseMirrorVerb().execute(ctx, collapse_request))

    llm_channel, llm_envelope = bus.calls[-1]
    assert "LLMGatewayService" in llm_channel
    prompt = llm_envelope.payload["messages"][1]["content"]
    assert "SELF-STUDY CONTEXT mode=reflective consumer=actions.respond_to_juniper_collapse_mirror.v1 policy=policy_allowed" in prompt
    assert "[reflective|reflection]" in prompt
    assert output.metadata["self_study"]["result"]["groups"][2]["items"][0]["trust_tier"] == "reflective"
    assert output.metadata["self_study"]["result"]["groups"][2]["items"][0]["metadata"]["provenance"] == ["repo"]
    assert output.recall_debug["self_study"]["policy_decision"]["allowed_trust_tiers"] == [
        "authoritative",
        "induced",
        "reflective",
    ]
