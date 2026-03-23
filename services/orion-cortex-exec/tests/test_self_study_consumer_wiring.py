import asyncio
import os
import sys
from uuid import uuid4

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.verb_adapters import LegacyPlanVerb  # noqa: E402
from orion.core.bus.bus_schemas import ServiceRef  # noqa: E402
from orion.core.verbs.base import VerbContext  # noqa: E402
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest, PlanExecutionResult  # noqa: E402
from orion.schemas.self_study import SelfStudyRetrieveResultV1  # noqa: E402


def _payload(*, output_mode: str, self_study: dict | None = None) -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="write_guide",
            steps=[],
        ),
        args=PlanExecutionArgs(
            request_id=str(uuid4()),
            extra={
                "mode": "brain",
                "options": {"self_study": self_study or {}},
            },
        ),
        context={
            "messages": [{"role": "user", "content": "help me with the Orion repo"}],
            "output_mode": output_mode,
            "metadata": {},
        },
    )


def test_legacy_plan_delivery_self_study_downgrades_to_factual(monkeypatch):
    captured = {}

    async def _fake_self_retrieve(*, request, **_kwargs):
        captured["request"] = request
        return SelfStudyRetrieveResultV1.model_validate(
            {
                "run_id": "run-1",
                "retrieval_mode": "factual",
                "applied_filters": request.filters.model_dump(),
                "groups": [
                    {
                        "trust_tier": "authoritative",
                        "items": [
                            {
                                "stable_id": "fact-1",
                                "trust_tier": "authoritative",
                                "record_type": "fact",
                                "title": "Exec bus runtime",
                                "content_preview": "Verb adapters run inside cortex-exec.",
                                "source_kind": "self_study",
                                "source_snapshot_id": "snapshot-1",
                                "source_path": "services/orion-cortex-exec/app/verb_adapters.py",
                                "evidence": [],
                                "concept_refs": [],
                                "metadata": {},
                            }
                        ],
                    }
                ],
                "counts": {"total": 1, "authoritative": 1, "induced": 0, "reflective": 0, "facts": 1, "concepts": 0, "reflections": 0},
                "backend_status": [],
                "notes": [],
            }
        )

    async def _fake_run_plan(self, bus, *, source, req, correlation_id, ctx):
        captured["ctx"] = ctx
        return PlanExecutionResult(
            verb_name=req.plan.verb_name,
            request_id=req.args.request_id,
            status="success",
            steps=[],
            mode="brain",
            final_text="ok",
            memory_used=False,
            recall_debug={},
            error=None,
        )

    monkeypatch.setattr("app.verb_adapters.run_self_retrieve", _fake_self_retrieve)
    monkeypatch.setattr("app.verb_adapters.PlanRouter.run_plan", _fake_run_plan)

    ctx = VerbContext(meta={"bus": object(), "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": str(uuid4())})
    payload = _payload(output_mode="implementation_guide", self_study={"enabled": True, "retrieval_mode": "reflective"})

    output, _effects = asyncio.run(LegacyPlanVerb().execute(ctx, payload))

    assert captured["request"].retrieval_mode == "factual"
    assert "SELF-STUDY CONTEXT mode=factual consumer=legacy.plan policy=policy_downgraded_to_factual" in captured["ctx"]["self_study_rendered"]
    assert captured["ctx"]["self_study"]["result"]["counts"]["reflective"] == 0
    assert captured["ctx"]["messages"][-1]["role"] == "system"
    assert output.result.recall_debug["self_study"]["policy_decision"]["allowed_trust_tiers"] == ["authoritative"]


def test_legacy_plan_planning_self_study_preserves_conceptual_metadata(monkeypatch):
    captured = {}

    async def _fake_self_retrieve(*, request, **_kwargs):
        return SelfStudyRetrieveResultV1.model_validate(
            {
                "run_id": "run-2",
                "retrieval_mode": "conceptual",
                "applied_filters": request.filters.model_dump(),
                "groups": [
                    {
                        "trust_tier": "authoritative",
                        "items": [
                            {
                                "stable_id": "fact-1",
                                "trust_tier": "authoritative",
                                "record_type": "fact",
                                "title": "Planner surface",
                                "content_preview": "PlannerReact is used for agent depth.",
                                "source_kind": "self_study",
                                "source_snapshot_id": "snapshot-1",
                                "source_path": "services/orion-cortex-exec/app/executor.py",
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
                                "title": "Planning cluster",
                                "content_preview": "Planner and agent chain coordinate execution.",
                                "source_kind": "self_study",
                                "source_snapshot_id": "snapshot-1",
                                "source_path": "services/orion-cortex-exec/app/router.py",
                                "evidence": [],
                                "concept_refs": [],
                                "metadata": {"provenance": ["repo"]},
                            }
                        ],
                    },
                ],
                "counts": {"total": 2, "authoritative": 1, "induced": 1, "reflective": 0, "facts": 1, "concepts": 1, "reflections": 0},
                "backend_status": [],
                "notes": [],
            }
        )

    async def _fake_run_plan(self, bus, *, source, req, correlation_id, ctx):
        captured["ctx"] = ctx
        return PlanExecutionResult(
            verb_name=req.plan.verb_name,
            request_id=req.args.request_id,
            status="success",
            steps=[],
            mode="brain",
            final_text="ok",
            memory_used=False,
            recall_debug={},
            error=None,
        )

    monkeypatch.setattr("app.verb_adapters.run_self_retrieve", _fake_self_retrieve)
    monkeypatch.setattr("app.verb_adapters.PlanRouter.run_plan", _fake_run_plan)

    ctx = VerbContext(meta={"bus": object(), "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": str(uuid4())})
    payload = _payload(output_mode="project_planning", self_study={"enabled": True, "retrieval_mode": "conceptual"})

    output, _effects = asyncio.run(LegacyPlanVerb().execute(ctx, payload))

    self_study = output.result.recall_debug["self_study"]
    assert self_study["retrieval_mode"] == "conceptual"
    assert self_study["policy_decision"]["consumer_kind"] == "planning_architecture"
    assert self_study["result"]["groups"][1]["items"][0]["trust_tier"] == "induced"
    assert self_study["result"]["groups"][1]["items"][0]["metadata"]["provenance"] == ["repo"]
    assert "induced" in captured["ctx"]["self_study_rendered"]


def test_legacy_plan_self_study_unavailable_is_graceful(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("backend_down")

    async def _fake_run_plan(self, bus, *, source, req, correlation_id, ctx):
        return PlanExecutionResult(
            verb_name=req.plan.verb_name,
            request_id=req.args.request_id,
            status="success",
            steps=[],
            mode="brain",
            final_text="ok",
            memory_used=False,
            recall_debug={},
            error=None,
        )

    monkeypatch.setattr("app.verb_adapters.run_self_retrieve", _boom)
    monkeypatch.setattr("app.verb_adapters.PlanRouter.run_plan", _fake_run_plan)

    ctx = VerbContext(meta={"bus": object(), "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": str(uuid4())})
    payload = _payload(output_mode="project_planning", self_study={"enabled": True, "retrieval_mode": "conceptual"})

    output, _effects = asyncio.run(LegacyPlanVerb().execute(ctx, payload))

    self_study = output.result.recall_debug["self_study"]
    assert self_study["used"] is False
    assert "self_study_unavailable:backend_down" in self_study["notes"]
    assert self_study["rendered"] == "SELF-STUDY CONTEXT consumer=legacy.plan status=disabled reason=policy_allowed."
