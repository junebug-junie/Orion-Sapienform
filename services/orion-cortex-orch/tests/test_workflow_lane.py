from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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
from orion.spark.introspection_metadata import build_introspection_context
from orion.spark.concept_induction.parity_evidence import (
    ParityReadinessThresholds,
    configure_parity_evidence_store,
    record_parity_evidence,
    reset_parity_evidence_store,
)


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


def test_orch_info_request_returns_runtime_identity() -> None:
    env = BaseEnvelope(
        kind='orion.cortex.orch.info.request.v1',
        source=ServiceRef(name='cortex-gateway'),
        correlation_id='00000000-0000-0000-0000-000000000000',
        payload={},
    )
    res = asyncio.run(orch_main.handle(env))
    assert res.kind == 'orion.cortex.orch.info.result.v1'
    assert res.payload['service']
    assert res.payload['version']
    assert res.payload['process_started_at']


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
    trigger_envelopes = [env for channel, env in bus.published if channel == "orion:actions:trigger:workflow.v1"]
    assert trigger_envelopes
    dispatch_payload = trigger_envelopes[-1].payload
    assert dispatch_payload["execution_policy"]["invocation_mode"] == "scheduled"
    assert dispatch_payload["execution_policy"]["schedule"]["kind"] == "one_shot"


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


def test_dream_cycle_reports_persisted_only_when_confirmed(monkeypatch) -> None:
    async def _fake_call_verb_runtime(*args, **kwargs):
        return DummyVerbResult(
            payload={
                'result': {
                    'status': 'success',
                    'final_text': 'Dream synthesis complete.',
                    'steps': [],
                    'memory_used': True,
                    'recall_debug': {'profile': 'dream.v1'},
                    'metadata': {'dream_persisted': True},
                }
            }
        )

    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('dream_cycle'),
            correlation_id='00000000-0000-0000-0000-000000000099',
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )
    assert result.ok is True
    assert result.metadata['workflow']['persisted'] == ['dream.result.v1']
    assert result.metadata['workflow']['dream_persistence_confirmed'] is True


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
    assert result.metadata['workflow']['journal_entry']['title']
    assert result.metadata['workflow']['journal_entry']['body'] == 'Meaningful Body'
    assert any(channel == 'orion:journal:write' for channel, _ in bus.published)


def test_journal_pass_primary_metadata_continuity(monkeypatch) -> None:
    seen_metadata: dict = {}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs['client_request']
        seen_metadata.update(req.context.metadata or {})
        return DummyVerbResult(payload={'result': {'status': 'success', 'final_text': '{"mode":"manual","title":"Title","body":"Meaningful Body"}', 'steps': [], 'metadata': {}, 'recall_debug': {}}})

    asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('journal_pass'),
            correlation_id='00000000-0000-0000-0000-000000010001',
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )
    assert seen_metadata.get("workflow_id") == "journal_pass"
    assert seen_metadata.get("workflow_execution", {}).get("workflow_subverb") == "journal.compose"
    assert seen_metadata.get("session_id") == "sid-1"
    assert seen_metadata.get("user_id") == "user-1"


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


def test_self_review_primary_metadata_continuity(monkeypatch) -> None:
    seen_metadata: dict = {}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs['client_request']
        seen_metadata.update(req.context.metadata or {})
        return DummyVerbResult(payload={'result': {'status': 'success', 'final_text': '{"summary":"Reflective summary","findings":[]}', 'steps': [], 'metadata': {'skill_result': {'summary': 'Reflective summary', 'findings': []}}}})

    asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('self_review'),
            correlation_id='00000000-0000-0000-0000-000000010002',
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )
    assert seen_metadata.get("workflow_id") == "self_review"
    assert seen_metadata.get("workflow_execution", {}).get("workflow_subverb") == "self_concept_reflect"
    assert seen_metadata.get("trace_id") == "trace-1"


def test_introspection_parity_for_primary_workflow_metadata() -> None:
    spark_meta = {
        "session_id": "sid-1",
        "user_id": "user-1",
        "workflow_id": "self_review",
        "trace_verb": "self_concept_reflect",
        "personality_file": "orion/cognition/personality/orion_identity.yaml",
    }
    continuity = build_introspection_context(
        spark_meta=spark_meta,
        trace_id="trace-1",
        correlation_id="corr-1",
    )
    assert continuity["session_id"] == "sid-1"
    assert continuity["user_id"] == "user-1"
    assert continuity["workflow_id"] == "self_review"
    assert continuity["personality_file"] == "orion/cognition/personality/orion_identity.yaml"


def test_self_review_usefulness_for_zero_and_nonzero_findings(monkeypatch) -> None:
    async def _zero_findings(*args, **kwargs):
        return DummyVerbResult(payload={'result': {'status': 'success', 'final_text': '{"summary":"","findings":[]}', 'steps': [], 'metadata': {'skill_result': {'summary': '', 'findings': []}}}})

    zero_result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('self_review'),
            correlation_id='00000000-0000-0000-0000-000000010003',
            causality_chain=[],
            trace={},
            call_verb_runtime=_zero_findings,
        )
    )
    assert "No notable self-review findings were identified" in (zero_result.final_text or "")

    async def _with_findings(*args, **kwargs):
        return DummyVerbResult(payload={'result': {'status': 'success', 'final_text': '{"summary":"ignored","findings":[{"kind":"seam_risk"},{"kind":"alignment_gap"}]}', 'steps': [], 'metadata': {'skill_result': {'summary': 'ignored', 'findings': [{'kind': 'seam_risk'}, {'kind': 'alignment_gap'}]}}}})

    findings_result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('self_review'),
            correlation_id='00000000-0000-0000-0000-000000010004',
            causality_chain=[],
            trace={},
            call_verb_runtime=_with_findings,
        )
    )
    assert "Self review completed with 2 findings" in (findings_result.final_text or "")


def test_concept_induction_pass_reviews_existing_profiles(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime
    from orion.spark.concept_induction.profile_repository import build_concept_profile_repository

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / 'concepts.json')
        subjects = ['orion']

    (tmp_path / 'concepts.json').write_text('{"profiles": {"orion": {"profile_id": "profile-1", "subject": "orion", "revision": 2, "created_at": "2026-03-23T00:00:00+00:00", "window_start": "2026-03-22T00:00:00+00:00", "window_end": "2026-03-23T00:00:00+00:00", "concepts": [{"concept_id": "concept-1", "label": "continuity", "aliases": [], "type": "identity", "salience": 1.0, "confidence": 0.8, "embedding_ref": null, "evidence": [], "metadata": {}}], "clusters": [{"cluster_id": "cluster-1", "label": "core", "summary": "core cluster", "concept_ids": ["concept-1"], "cohesion_score": 0.8, "metadata": {}}], "state_estimate": null, "metadata": {}}}}')
    monkeypatch.setattr(workflow_runtime, 'build_orch_concept_profile_settings', lambda *_args, **_kwargs: FakeConceptSettings())

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
    assert result.metadata['workflow']['profile_store_path'] == str(tmp_path / 'concepts.json')
    assert result.metadata['workflow']['profiles_reviewed'][0]['subject'] == 'orion'
    details = result.metadata["workflow"]["concept_induction_details"]
    assert details["profiles"][0]["profile_id"] == "profile-1"
    assert details["profiles"][0]["concepts"][0]["label"] == "continuity"
    assert "trace" in details
    assert build_concept_profile_repository(FakeConceptSettings()).status().backend == 'local'


def test_concept_induction_pass_synthesize_to_journal_uses_append_only_write(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / 'concepts.json')
        subjects = ['orion']

    (tmp_path / 'concepts.json').write_text('{"profiles": {"orion": {"profile_id": "profile-1", "subject": "orion", "revision": 4, "created_at": "2026-03-23T00:00:00+00:00", "window_start": "2026-03-22T00:00:00+00:00", "window_end": "2026-03-23T00:00:00+00:00", "concepts": [{"concept_id": "concept-1", "label": "continuity", "aliases": [], "type": "identity", "salience": 1.0, "confidence": 0.8, "embedding_ref": null, "evidence": [], "metadata": {}}], "clusters": [{"cluster_id": "cluster-1", "label": "core", "summary": "core cluster", "concept_ids": ["concept-1"], "cohesion_score": 0.8, "metadata": {}}], "state_estimate": null, "metadata": {}}}}')
    monkeypatch.setattr(workflow_runtime, 'build_orch_concept_profile_settings', lambda *_args, **_kwargs: FakeConceptSettings())
    req = _req("concept_induction_pass")
    req.context.metadata["workflow_request"]["action"] = "synthesize_to_journal"
    bus = DummyBus()
    result = asyncio.run(
        execute_chat_workflow(
            bus=bus,
            source=ServiceRef(name='cortex-orch'),
            req=req,
            correlation_id='00000000-0000-0000-0000-000000000095',
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert result.ok is True
    assert result.metadata["workflow"]["persisted"]
    assert result.metadata["workflow"]["synthesis_to_journal"]["ok"] is True
    assert result.metadata["workflow"]["synthesis_to_journal"]["journal_entry"]["source_ref"].startswith("concept_induction_pass:")
    assert any(channel == "orion:journal:write" for channel, _ in bus.published)
    payload = result.metadata["workflow"]["synthesis_to_journal"]["provenance"]
    assert payload["source_workflow_id"] == "concept_induction_pass"
    assert payload["reviewed_profiles"][0]["profile_id"] == "profile-1"


def test_concept_induction_pass_uses_repository_seam(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / 'concepts.json')
        subjects = ['orion']

    class FakeRepository:
        def __init__(self) -> None:
            self.list_latest_called = False
            self.last_observer = None

        def status(self):
            return SimpleNamespace(
                backend='local',
                source_path=str(tmp_path / 'concepts.json'),
                placeholder_default_in_use=False,
                source_available=True,
            )

        def list_latest(self, subjects, *, observer=None):
            self.list_latest_called = True
            self.last_observer = observer
            return []

    fake_repository = FakeRepository()
    monkeypatch.setattr(workflow_runtime, 'build_orch_concept_profile_settings', lambda *_args, **_kwargs: FakeConceptSettings())
    monkeypatch.setattr(workflow_runtime, 'build_concept_profile_repository', lambda settings, backend_override=None: fake_repository)
    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('concept_induction_pass'),
            correlation_id='00000000-0000-0000-0000-000000000015',
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert fake_repository.list_latest_called is True
    assert fake_repository.last_observer["consumer"] == "concept_induction_pass"
    assert fake_repository.last_observer["correlation_id"] == "00000000-0000-0000-0000-000000000015"
    assert result.ok is False


def test_concept_induction_pass_graph_override_uses_graph_when_available(monkeypatch, tmp_path, caplog) -> None:
    from app import workflow_runtime

    caplog.set_level("INFO")

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / "concepts.json")
        subjects = ["orion"]
        concept_profile_repository_backend = "local"
        concept_profile_backend_concept_induction_pass = "graph"
        concept_profile_graph_cutover_fallback_policy = "fail_open_local"

    class FakeRepository:
        def __init__(self, backend: str):
            self.backend = backend
            self.observer = None

        def status(self):
            return SimpleNamespace(
                backend=self.backend,
                source_path=str(tmp_path / "concepts.json"),
                placeholder_default_in_use=False,
                source_available=True,
            )

        def list_latest(self, subjects, *, observer=None):
            self.observer = observer
            profile = SimpleNamespace(
                profile_id="profile-graph",
                subject="orion",
                revision=3,
                concepts=[SimpleNamespace(label="continuity")],
                clusters=[],
                state_estimate=None,
                window_start=datetime.now(timezone.utc),
                window_end=datetime.now(timezone.utc),
            )
            return [SimpleNamespace(subject="orion", profile=profile, availability="available", unavailable_reason=None)]

    graph_repo = FakeRepository("graph")
    monkeypatch.setattr(workflow_runtime, "build_orch_concept_profile_settings", lambda *_args, **_kwargs: FakeConceptSettings())
    monkeypatch.setattr(
        workflow_runtime,
        "build_concept_profile_repository",
        lambda settings, backend_override=None: graph_repo,
    )

    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name="cortex-orch"),
            req=_req("concept_induction_pass"),
            correlation_id="00000000-0000-0000-0000-000000000099",
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )

    assert result.ok is True
    assert result.metadata["workflow"]["profiles_reviewed"][0]["profile_id"] == "profile-graph"
    resolution = result.metadata["workflow"]["concept_profile_resolution"]
    assert resolution["requested_backend"] == "graph"
    assert resolution["resolved_backend"] == "graph"
    assert resolution["fallback_used"] is False
    assert graph_repo.observer["consumer"] == "concept_induction_pass"
    assert "concept_profile_repository_resolution" in caplog.text


def test_concept_induction_pass_graph_unavailable_fail_open_falls_back_to_local(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / "concepts.json")
        subjects = ["orion"]
        concept_profile_repository_backend = "local"
        concept_profile_backend_concept_induction_pass = "graph"
        concept_profile_graph_cutover_fallback_policy = "fail_open_local"

    class FakeRepository:
        def __init__(self, backend: str):
            self.backend = backend

        def status(self):
            return SimpleNamespace(
                backend=self.backend,
                source_path=str(tmp_path / "concepts.json"),
                placeholder_default_in_use=False,
                source_available=True,
            )

        def list_latest(self, subjects, *, observer=None):
            if self.backend == "graph":
                return [SimpleNamespace(subject="orion", profile=None, availability="unavailable", unavailable_reason="query_error")]
            profile = SimpleNamespace(
                profile_id="profile-local",
                subject="orion",
                revision=2,
                concepts=[SimpleNamespace(label="continuity")],
                clusters=[],
                state_estimate=None,
                window_start=datetime.now(timezone.utc),
                window_end=datetime.now(timezone.utc),
            )
            return [SimpleNamespace(subject="orion", profile=profile, availability="available", unavailable_reason=None)]

    build_calls: list[str] = []

    def _build(settings, backend_override=None):
        backend = backend_override or settings.concept_profile_repository_backend
        build_calls.append(backend)
        return FakeRepository(backend)

    monkeypatch.setattr(workflow_runtime, "build_orch_concept_profile_settings", lambda *_args, **_kwargs: FakeConceptSettings())
    monkeypatch.setattr(workflow_runtime, "build_concept_profile_repository", _build)

    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name="cortex-orch"),
            req=_req("concept_induction_pass"),
            correlation_id="00000000-0000-0000-0000-000000000100",
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert result.ok is True
    assert build_calls == ["graph", "local"]
    assert result.metadata["workflow"]["profiles_reviewed"][0]["profile_id"] == "profile-local"
    resolution = result.metadata["workflow"]["concept_profile_resolution"]
    assert resolution["requested_backend"] == "graph"
    assert resolution["resolved_backend"] == "local"
    assert resolution["fallback_used"] is True
    assert resolution["unavailable_reason"] == "query_error"


def test_concept_induction_pass_graph_unavailable_fail_closed_fails(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / "concepts.json")
        subjects = ["orion"]
        concept_profile_repository_backend = "local"
        concept_profile_backend_concept_induction_pass = "graph"
        concept_profile_graph_cutover_fallback_policy = "fail_closed"

    class FakeGraphRepository:
        def status(self):
            return SimpleNamespace(
                backend="graph",
                source_path="graphdb://collapse",
                placeholder_default_in_use=False,
                source_available=True,
            )

        def list_latest(self, subjects, *, observer=None):
            return [SimpleNamespace(subject="orion", profile=None, availability="unavailable", unavailable_reason="graph_not_configured")]

    monkeypatch.setattr(workflow_runtime, "build_orch_concept_profile_settings", lambda *_args, **_kwargs: FakeConceptSettings())
    monkeypatch.setattr(
        workflow_runtime,
        "build_concept_profile_repository",
        lambda settings, backend_override=None: FakeGraphRepository(),
    )
    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name="cortex-orch"),
            req=_req("concept_induction_pass"),
            correlation_id="00000000-0000-0000-0000-000000000101",
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert result.ok is False
    assert result.error["code"] == "concept_profiles_graph_unavailable"
    resolution = result.metadata["workflow"]["concept_profile_resolution"]
    assert resolution["requested_backend"] == "graph"
    assert resolution["resolved_backend"] == "graph"
    assert resolution["fallback_used"] is False
    assert resolution["unavailable_reason"] == "graph_not_configured"


def test_concept_induction_pass_local_override_behaves_like_local(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / "concepts.json")
        subjects = ["orion"]
        concept_profile_repository_backend = "graph"
        concept_profile_backend_concept_induction_pass = "local"
        concept_profile_graph_cutover_fallback_policy = "fail_closed"

    class FakeLocalRepository:
        def status(self):
            return SimpleNamespace(backend="local", source_path=str(tmp_path / "concepts.json"), placeholder_default_in_use=False, source_available=True)

        def list_latest(self, subjects, *, observer=None):
            profile = SimpleNamespace(
                profile_id="profile-local",
                subject="orion",
                revision=1,
                concepts=[SimpleNamespace(label="identity")],
                clusters=[],
                state_estimate=None,
                window_start=datetime.now(timezone.utc),
                window_end=datetime.now(timezone.utc),
            )
            return [SimpleNamespace(subject="orion", profile=profile, availability="available", unavailable_reason=None)]

    chosen_backends: list[str] = []
    monkeypatch.setattr(workflow_runtime, "build_orch_concept_profile_settings", lambda *_args, **_kwargs: FakeConceptSettings())
    monkeypatch.setattr(
        workflow_runtime,
        "build_concept_profile_repository",
        lambda settings, backend_override=None: (chosen_backends.append(backend_override or settings.concept_profile_repository_backend), FakeLocalRepository())[1],
    )
    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name="cortex-orch"),
            req=_req("concept_induction_pass"),
            correlation_id="00000000-0000-0000-0000-000000000102",
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert result.ok is True
    assert chosen_backends == ["local"]
    assert result.metadata["workflow"]["concept_profile_resolution"]["resolved_backend"] == "local"


def test_concept_induction_pass_shadow_override_retains_shadow_behavior(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / "concepts.json")
        subjects = ["orion"]
        concept_profile_repository_backend = "local"
        concept_profile_backend_concept_induction_pass = "shadow"
        concept_profile_graph_cutover_fallback_policy = "fail_open_local"

    class FakeShadowRepository:
        def __init__(self):
            self.observer = None

        def status(self):
            return SimpleNamespace(backend="shadow", source_path=str(tmp_path / "concepts.json"), placeholder_default_in_use=False, source_available=True)

        def list_latest(self, subjects, *, observer=None):
            self.observer = observer
            profile = SimpleNamespace(
                profile_id="profile-shadow-local",
                subject="orion",
                revision=4,
                concepts=[SimpleNamespace(label="continuity")],
                clusters=[],
                state_estimate=None,
                window_start=datetime.now(timezone.utc),
                window_end=datetime.now(timezone.utc),
            )
            return [SimpleNamespace(subject="orion", profile=profile, availability="available", unavailable_reason=None)]

    fake_repo = FakeShadowRepository()
    monkeypatch.setattr(workflow_runtime, "build_orch_concept_profile_settings", lambda *_args, **_kwargs: FakeConceptSettings())
    monkeypatch.setattr(workflow_runtime, "build_concept_profile_repository", lambda settings, backend_override=None: fake_repo)
    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name="cortex-orch"),
            req=_req("concept_induction_pass"),
            correlation_id="00000000-0000-0000-0000-000000000103",
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert result.ok is True
    assert result.metadata["workflow"]["profiles_reviewed"][0]["profile_id"] == "profile-shadow-local"
    assert result.metadata["workflow"]["concept_profile_resolution"]["resolved_backend"] == "shadow"
    assert fake_repo.observer["consumer"] == "concept_induction_pass"


def test_concept_profile_backend_override_does_not_change_chat_stance_resolution(monkeypatch) -> None:
    from app import workflow_runtime

    settings = SimpleNamespace(
        concept_profile_repository_backend="shadow",
        concept_profile_backend_concept_induction_pass="graph",
    )
    concept_backend = workflow_runtime._resolve_concept_profile_backend_for_consumer(
        settings,
        consumer="concept_induction_pass",
    )
    chat_backend = workflow_runtime._resolve_concept_profile_backend_for_consumer(
        settings,
        consumer="chat_stance",
    )
    assert concept_backend == "graph"
    assert chat_backend == "shadow"


def test_concept_induction_pass_rollback_from_graph_override_restores_global(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / "concepts.json")
        subjects = ["orion"]
        concept_profile_repository_backend = "local"
        concept_profile_backend_concept_induction_pass = ""
        concept_profile_graph_cutover_fallback_policy = "fail_open_local"

    class FakeLocalRepository:
        def status(self):
            return SimpleNamespace(backend="local", source_path=str(tmp_path / "concepts.json"), placeholder_default_in_use=False, source_available=True)

        def list_latest(self, subjects, *, observer=None):
            return []

    selected: list[str] = []
    monkeypatch.setattr(workflow_runtime, "build_orch_concept_profile_settings", lambda *_args, **_kwargs: FakeConceptSettings())
    monkeypatch.setattr(
        workflow_runtime,
        "build_concept_profile_repository",
        lambda settings, backend_override=None: (selected.append(backend_override or settings.concept_profile_repository_backend), FakeLocalRepository())[1],
    )
    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name="cortex-orch"),
            req=_req("concept_induction_pass"),
            correlation_id="00000000-0000-0000-0000-000000000104",
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert selected == ["local"]
    assert result.metadata["workflow"]["concept_profile_resolution"]["requested_backend"] == "local"
    assert result.metadata["workflow"]["concept_profile_resolution"]["fallback_used"] is False


def test_concept_induction_pass_fails_honestly_when_profiles_missing(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = str(tmp_path / 'concepts.json')
        subjects = ['orion']

    monkeypatch.setattr(workflow_runtime, 'build_orch_concept_profile_settings', lambda *_args, **_kwargs: FakeConceptSettings())
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


def test_concept_induction_pass_reports_placeholder_store_as_unconfigured(monkeypatch) -> None:
    from app import workflow_runtime

    class FakeConceptSettings(SimpleNamespace):
        store_path = '/tmp/concept-induction-state.json'
        subjects = ['orion']

    monkeypatch.setattr(workflow_runtime, 'build_orch_concept_profile_settings', lambda *_args, **_kwargs: FakeConceptSettings())
    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('concept_induction_pass'),
            correlation_id='00000000-0000-0000-0000-000000000007',
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )
    assert result.ok is False
    assert result.metadata['workflow']['profile_store_placeholder_path'] is True
    assert 'Set CONCEPT_STORE_PATH to a real profile store path' in result.metadata['workflow']['main_result']


def test_orch_handle_workflow_failure_returns_explicit_failure_without_chat_fallback(monkeypatch) -> None:
    async def _raise_execute_chat_workflow(**kwargs):
        raise RuntimeError("workflow backend unavailable")

    monkeypatch.setattr(orch_main, "execute_chat_workflow", _raise_execute_chat_workflow)

    env = BaseEnvelope(
        kind="cortex.orch.request",
        source=ServiceRef(name="cortex-gateway"),
        correlation_id="00000000-0000-0000-0000-000000009001",
        payload=_req("journal_pass").model_dump(mode="json"),
    )
    res = asyncio.run(orch_main.handle(env))
    payload = CortexClientResult.model_validate(res.payload)
    assert payload.ok is False
    assert payload.verb == "journal_pass"
    assert "not replaced with chat_general" in (payload.final_text or "")


def test_orch_info_surface_includes_concept_profile_parity_evidence() -> None:
    reset_parity_evidence_store()
    configure_parity_evidence_store(
        thresholds=ParityReadinessThresholds(min_comparisons=1, max_mismatch_rate=1.0, max_unavailable_rate=1.0),
        summary_interval=1,
    )
    record_parity_evidence(
        consumer="concept_induction_pass",
        subject_outcomes=[
            {"subject": "orion", "mismatch_classes": [], "graph_unavailable": False, "empty_on_local_only": False, "empty_on_graph_only": False}
        ],
    )
    env = BaseEnvelope(
        kind="orion.cortex.orch.info.request.v1",
        source=ServiceRef(name="test"),
        correlation_id="00000000-0000-0000-0000-000000009777",
        payload={},
    )
    res = asyncio.run(orch_main.handle(env))
    assert "concept_profile_parity_evidence" in res.payload
    parity = res.payload["concept_profile_parity_evidence"]
    assert "consumers" in parity
    assert parity["consumers"]["concept_induction_pass"]["total_comparisons"] == 1


def test_workflow_summary_never_claims_persisted_without_confirmed_write() -> None:
    async def _fake_call_verb_runtime(*args, **kwargs):
        return DummyVerbResult(
            payload={
                "result": {
                    "status": "success",
                    "final_text": "Dream synthesis complete.",
                    "steps": [],
                    "memory_used": True,
                    "recall_debug": {"profile": "dream.v1"},
                    "metadata": {},
                }
            }
        )

    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name="cortex-orch"),
            req=_req("dream_cycle"),
            correlation_id="00000000-0000-0000-0000-000000009002",
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )
    assert "Persisted: none" in (result.final_text or "")
    assert result.metadata["workflow"]["persisted"] == []


def test_dream_cycle_primary_metadata_continuity(monkeypatch) -> None:
    seen_metadata: dict = {}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs['client_request']
        seen_metadata.update(req.context.metadata or {})
        return DummyVerbResult(
            payload={
                'result': {
                    'status': 'success',
                    'final_text': 'Dream synthesis complete.',
                    'steps': [],
                    'memory_used': True,
                    'recall_debug': {'profile': 'dream.v1'},
                    'metadata': {},
                }
            }
        )

    asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name='cortex-orch'),
            req=_req('dream_cycle'),
            correlation_id='00000000-0000-0000-0000-000000010005',
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )
    assert seen_metadata.get("workflow_id") == "dream_cycle"
    assert seen_metadata.get("workflow_execution", {}).get("workflow_subverb") == "dream_cycle"


def test_concept_induction_pass_does_not_pass_generic_orch_settings_to_repository_builder(monkeypatch, tmp_path) -> None:
    from app import workflow_runtime
    from app.settings import Settings as OrchSettings

    class FakeConceptProfileSettings(SimpleNamespace):
        store_path = str(tmp_path / "concepts.json")
        subjects = ["orion"]
        concept_profile_repository_backend = "graph"
        concept_profile_backend_concept_induction_pass = "graph"
        concept_profile_graph_cutover_fallback_policy = "fail_open_local"
        concept_profile_graphdb_endpoint = "http://graphdb:7200/repositories/collapse"
        concept_profile_graph_uri = "http://example.org/graph/concept-profile"
        concept_profile_graph_timeout_sec = 1.0

    class FakeRepository:
        def status(self):
            return SimpleNamespace(
                backend="graph",
                source_path="graphdb://collapse",
                placeholder_default_in_use=False,
                source_available=True,
            )

        def list_latest(self, subjects, *, observer=None):
            return [SimpleNamespace(subject="orion", profile=None, availability="unavailable", unavailable_reason="query_error")]

    orch_settings = OrchSettings()
    captured_settings_types: list[type] = []

    def _build(settings, backend_override=None):
        captured_settings_types.append(type(settings))
        assert not isinstance(settings, OrchSettings)
        assert hasattr(settings, "store_path")
        return FakeRepository()

    monkeypatch.setattr(workflow_runtime, "get_settings", lambda: orch_settings)
    monkeypatch.setattr(workflow_runtime, "build_orch_concept_profile_settings", lambda orch: FakeConceptProfileSettings())
    monkeypatch.setattr(workflow_runtime, "build_concept_profile_repository", _build)

    result = asyncio.run(
        execute_chat_workflow(
            bus=DummyBus(),
            source=ServiceRef(name="cortex-orch"),
            req=_req("concept_induction_pass"),
            correlation_id="00000000-0000-0000-0000-000000000105",
            causality_chain=[],
            trace={},
            call_verb_runtime=lambda *args, **kwargs: None,
        )
    )

    assert captured_settings_types
    assert result.ok is False
