from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, APP_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientResult

from app import main as orch_main
from app.workflow_runtime import execute_chat_workflow


class DummyBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))


class DummyVerbResult:
    def __init__(self, *, ok: bool = True, payload: dict | None = None, error: str | None = None) -> None:
        self.ok = ok
        self.output = payload or {'result': {'status': 'success', 'final_text': 'ok', 'steps': [], 'metadata': {}}}
        self.error = error


def _req(workflow_id: str) -> CortexClientRequest:
    return CortexClientRequest.model_validate(
        {
            'mode': 'brain',
            'route_intent': 'none',
            'context': {
                'messages': [],
                'raw_user_text': 'workflow please',
                'user_message': 'workflow please',
                'session_id': 'sid-1',
                'user_id': 'user-1',
                'trace_id': 'trace-1',
                'metadata': {
                    'workflow_request': {
                        'workflow_id': workflow_id,
                        'matched_alias': workflow_id.replace('_', ' '),
                        'normalized_prompt': workflow_id,
                        'confidence': 1.0,
                        'resolver': 'alias_registry',
                        'invoked_from_chat': True,
                    }
                },
            },
            'options': {},
            'packs': [],
            'recall': {'enabled': True, 'required': False, 'mode': 'hybrid', 'profile': 'reflect.v1'},
        }
    )


def _req_with_policy(workflow_id: str, policy: dict) -> CortexClientRequest:
    req = _req(workflow_id)
    req.context.metadata["workflow_request"]["execution_policy"] = policy
    return req


def test_orch_handle_routes_explicit_workflow_request_before_normal_verb_runtime(monkeypatch) -> None:
    captured = {'workflow': 0, 'verb_runtime': 0}

    async def _fake_execute_chat_workflow(**kwargs):
        captured['workflow'] += 1
        return CortexClientResult(
            ok=True,
            mode='brain',
            verb='dream_cycle',
            status='success',
            final_text='workflow ok',
            memory_used=False,
            recall_debug={},
            steps=[],
            metadata={'workflow': {'workflow_id': 'dream_cycle'}},
        )

    async def _fake_call_verb_runtime(*args, **kwargs):
        captured['verb_runtime'] += 1
        raise AssertionError('normal verb runtime should not be used for explicit workflow requests')

    monkeypatch.setattr(orch_main, 'execute_chat_workflow', _fake_execute_chat_workflow)
    monkeypatch.setattr(orch_main, 'call_verb_runtime', _fake_call_verb_runtime)

    env = BaseEnvelope(
        kind='cortex.orch.request',
        source=ServiceRef(name='cortex-gateway'),
        correlation_id='00000000-0000-0000-0000-000000000001',
        payload=_req('dream_cycle').model_dump(mode='json'),
    )

    res = asyncio.run(orch_main.handle(env))

    assert captured['workflow'] == 1
    assert captured['verb_runtime'] == 0
    assert res.payload.metadata['workflow']['workflow_id'] == 'dream_cycle'


def test_scheduled_workflow_requests_publish_actions_trigger_instead_of_running_verb_runtime(monkeypatch) -> None:
    async def _fake_call_verb_runtime(*args, **kwargs):
        raise AssertionError("scheduled workflow should not execute runtime immediately")

    bus = DummyBus()
    req = _req_with_policy(
        "dream_cycle",
        {
            "workflow_id": "dream_cycle",
            "invocation_mode": "scheduled",
            "notify_on": "completion",
            "recipient_group": "juniper_primary",
            "schedule": {
                "kind": "one_shot",
                "timezone": "America/Denver",
                "run_at_utc": "2026-03-25T08:00:00Z",
                "label": "tomorrow morning",
            },
        },
    )
    result = asyncio.run(
        execute_chat_workflow(
            bus=bus,
            source=ServiceRef(name="cortex-orch"),
            req=req,
            correlation_id="00000000-0000-0000-0000-000000000101",
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )

    assert result.ok is True
    assert result.metadata["workflow"]["status"] == "scheduled"
    assert "Scheduled: none" not in result.final_text
    assert result.metadata["workflow"]["scheduled"]
    assert any(channel == "orion:actions:trigger:workflow.v1" for channel, _ in bus.published)


def test_dream_cycle_workflow_uses_existing_dream_verb(monkeypatch) -> None:
    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs['client_request']
        assert req.verb == 'dream_cycle'
        return DummyVerbResult(payload={'result': {'status': 'success', 'final_text': 'Dream synthesis complete.', 'steps': [], 'memory_used': True, 'recall_debug': {'profile': 'dream.v1'}, 'metadata': {}}})

    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('dream_cycle'),
            correlation_id='00000000-0000-0000-0000-000000000002',
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )

    assert result.ok is True
    assert result.metadata['workflow']['workflow_id'] == 'dream_cycle'
    assert result.metadata['workflow']['persisted'] == []
    assert result.metadata['workflow']['dream_persistence_confirmed'] is False
    assert 'Dream synthesis complete.' in result.final_text


def test_journal_pass_workflow_reuses_journal_compose_and_append_only_write(monkeypatch) -> None:
    bus = DummyBus()

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs['client_request']
        assert req.verb == 'journal.compose'
        return DummyVerbResult(payload={'result': {'status': 'success', 'final_text': '{"mode":"manual","title":null,"body":"Meaningful Body"}', 'steps': [], 'metadata': {}, 'recall_debug': {}}})

    result = asyncio.run(
        execute_chat_workflow(
            bus=bus,
            source=ServiceRef(name='cortex-orch'),
            req=_req('journal_pass'),
            correlation_id='00000000-0000-0000-0000-000000000003',
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )

    assert result.ok is True
    assert result.metadata['workflow']['workflow_id'] == 'journal_pass'
    assert result.metadata['workflow']['main_result'] is not None
    assert result.metadata['workflow']['main_result'] != 'None'
    assert any(channel == 'orion:journal:write' for channel, _ in bus.published)


def test_self_review_workflow_uses_existing_self_reflection_adapter(monkeypatch) -> None:
    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs['client_request']
        assert req.verb == 'self_concept_reflect'
        return DummyVerbResult(payload={'result': {'status': 'success', 'final_text': '{"summary":"Reflective summary","findings":[{"kind":"seam_risk"}],"graph_write":{"graph":"orion:self:reflective"},"journal_write":{"channel":"orion:journal:write"}}', 'steps': [], 'metadata': {'skill_result': {'summary': 'Reflective summary', 'findings': [{'kind': 'seam_risk'}], 'graph_write': {'graph': 'orion:self:reflective'}, 'journal_write': {'channel': 'orion:journal:write'}}}}})

    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('self_review'),
            correlation_id='00000000-0000-0000-0000-000000000004',
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )

    assert result.ok is True
    assert result.metadata['workflow']['workflow_id'] == 'self_review'
    assert result.metadata['workflow']['finding_count'] == 1


def test_concept_induction_pass_reviews_existing_profiles(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / 'concepts.json')
        subjects = ['orion']

    (tmp_path / 'concepts.json').write_text('{"profiles": {"orion": {"profile_id": "profile-1", "subject": "orion", "revision": 2, "created_at": "2026-03-23T00:00:00+00:00", "window_start": "2026-03-22T00:00:00+00:00", "window_end": "2026-03-23T00:00:00+00:00", "concepts": [{"concept_id": "concept-1", "label": "continuity", "aliases": [], "type": "identity", "salience": 1.0, "confidence": 0.8, "embedding_ref": null, "evidence": [], "metadata": {}}], "clusters": [{"cluster_id": "cluster-1", "label": "core", "summary": "core cluster", "concept_ids": ["concept-1"], "cohesion_score": 0.8, "metadata": {}}], "state_estimate": null, "metadata": {}}}}')
    monkeypatch.setattr(workflow_runtime, 'get_concept_settings', lambda: FakeConceptSettings())

    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('concept_induction_pass'),
            correlation_id='00000000-0000-0000-0000-000000000005',
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )

    assert result.ok is True
    assert result.metadata['workflow']['workflow_id'] == 'concept_induction_pass'
    assert result.metadata['workflow']['profiles_reviewed'][0]['subject'] == 'orion'


def test_concept_induction_pass_fails_honestly_when_profiles_missing(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / 'concepts.json')
        subjects = ['orion']

    monkeypatch.setattr(workflow_runtime, 'get_concept_settings', lambda: FakeConceptSettings())
    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('concept_induction_pass'),
            correlation_id='00000000-0000-0000-0000-000000000006',
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert result.ok is False
    assert result.metadata['workflow']['status'] == 'failed'
    assert result.error['code'] == 'concept_profiles_unavailable'
