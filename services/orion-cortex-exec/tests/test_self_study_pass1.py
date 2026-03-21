import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from uuid import uuid4

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_cortex_exec"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.self_study", APP_DIR / "self_study.py")
self_study = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = self_study
spec.loader.exec_module(self_study)

spec_va = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.verb_adapters", APP_DIR / "verb_adapters.py")
verb_adapters = importlib.util.module_from_spec(spec_va)
assert spec_va and spec_va.loader
sys.modules[spec_va.name] = verb_adapters
spec_va.loader.exec_module(verb_adapters)

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.verbs.base import VerbContext  # noqa: E402
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest  # noqa: E402


class _FakeBus:
    def __init__(self, *, fail_channel: str | None = None) -> None:
        self.fail_channel = fail_channel
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        if self.fail_channel == channel:
            raise RuntimeError(f"publish_failed:{channel}")
        self.published.append((channel, envelope))


def _request(verb_name: str = "self_repo_inspect", *, extra: dict | None = None) -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name=verb_name, steps=[]),
        args=PlanExecutionArgs(request_id=str(uuid4()), extra=extra or {}),
        context={},
    )


def test_build_self_snapshot_returns_typed_authoritative_shape():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    assert snapshot.trust_tier == "authoritative"
    assert snapshot.snapshot_id.startswith("self-snapshot-")
    assert snapshot.run_id.startswith("self-run-")
    assert snapshot.counts.services > 0
    assert snapshot.counts.channels > 0
    assert snapshot.counts.verbs > 0
    assert all(item.trust_tier == "authoritative" for item in snapshot.services)
    assert all(item.item_id.startswith("self-item-") for item in snapshot.services)
    assert all(item.run_id == snapshot.run_id for item in snapshot.services)
    assert any(item.name == "orion:rdf:enqueue" for item in snapshot.channels)
    assert any(item.name == "self_repo_inspect" for item in snapshot.verbs)


def test_repeated_snapshot_keeps_stable_snapshot_and_item_ids():
    first = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    second = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    assert first.snapshot_id == second.snapshot_id
    assert first.run_id != second.run_id
    assert [item.item_id for item in first.services] == [item.item_id for item in second.services]


def test_graph_writeback_payload_stays_authoritative_only():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    req = self_study.build_self_study_rdf_request(snapshot)

    assert req.graph == "orion:self"
    assert 'authoritative' in (req.triples or '')
    assert 'induced' not in (req.triples or '')
    assert 'reflective' not in (req.triples or '')
    assert snapshot.run_id == req.payload["run_id"]


def test_authoritative_builder_rejects_cross_tier_contamination():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    contaminated = snapshot.model_copy(deep=True)
    contaminated.services[0].trust_tier = "reflective"

    try:
        self_study.build_self_study_rdf_request(contaminated)
    except ValueError as exc:
        assert "non_authoritative_item" in str(exc)
    else:
        raise AssertionError("expected authoritative guard to reject reflective item")


def test_journal_entry_is_concise_and_marked_non_authoritative():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    journal = self_study.build_self_study_journal_entry(snapshot, correlation_id="corr-1")

    assert journal.source_kind == "self_study"
    assert journal.mode == "manual"
    assert journal.title == "Self-study factual snapshot"
    assert "Journal is summary-only and not authoritative storage." in journal.body


def test_induce_self_concepts_produces_induced_evidence_backed_concepts():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    concepts = self_study.induce_self_concepts(snapshot)

    assert concepts
    assert all(concept.trust_tier == "induced" for concept in concepts)
    assert all(concept.source_snapshot_id == snapshot.snapshot_id for concept in concepts)
    assert all(concept.evidence for concept in concepts)
    assert all(ref.trust_tier == "authoritative" for concept in concepts for ref in concept.evidence)
    assert any(concept.concept_kind == "runtime_boundary" for concept in concepts)
    assert any(concept.concept_kind == "graph_surface" for concept in concepts)

    summary = self_study.build_self_concept_summary(concepts)
    assert summary.startswith(f"Concept induction produced {len(concepts)} induced architectural concepts")


def test_induced_concept_rdf_request_targets_induced_graph_only():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)

    req = self_study.build_self_concept_rdf_request(
        source_snapshot=snapshot,
        concepts=concepts,
        run_id="concept-run-1",
    )

    assert req.graph == "orion:self:induced"
    assert req.kind == "self.concepts.induced.v1"
    assert req.payload["run_id"] == "concept-run-1"
    assert req.payload["source_snapshot_id"] == snapshot.snapshot_id
    assert req.payload["trust_tier"] == "induced"
    assert "InducedSelfConcept" in (req.triples or "")
    assert "authoritative" not in (req.payload["trust_tier"])
    assert "supportedBy" in (req.triples or "")
    assert all(ref.item_id.startswith("self-item-") for concept in concepts for ref in concept.evidence)


def test_induced_concept_rdf_request_rejects_missing_evidence():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)
    broken = concepts[0].model_copy(deep=True)
    broken.evidence = []

    try:
        self_study.build_self_concept_rdf_request(
            source_snapshot=snapshot,
            concepts=[broken],
            run_id="concept-run-2",
        )
    except ValueError as exc:
        assert "concept_missing_evidence" in str(exc)
    else:
        raise AssertionError("expected induced concept builder to reject concept without evidence")


def test_phase2a_validation_confirms_recall_isolation_and_concept_idempotency():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)

    summary = self_study.validate_phase2a_induction(snapshot, concepts)

    assert "self.factual.v1 excludes induced/reflective trust tiers" in summary
    assert "stable concept identifiers" in summary


def test_induction_ignores_journal_text_as_evidence_input():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    journal = self_study.build_self_study_journal_entry(snapshot, correlation_id="corr-journal")
    concepts = self_study.induce_self_concepts(snapshot)

    assert "Journal is summary-only" in journal.body
    assert all(journal.body not in concept.description for concept in concepts)
    assert all(ref.source_path != journal.source_ref for concept in concepts for ref in concept.evidence)


def test_publish_self_concepts_skips_unchanged_repeated_run():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)
    self_study._LAST_CONCEPT_PUBLISH_KEY = None

    first_bus = _FakeBus()
    first = asyncio.run(
        self_study.publish_self_concept_artifacts(
            bus=first_bus,
            source=ServiceRef(name="orion-cortex-exec"),
            snapshot=snapshot,
            concepts=concepts,
            correlation_id="corr-concept-1",
        )
    )
    second_bus = _FakeBus()
    second = asyncio.run(
        self_study.publish_self_concept_artifacts(
            bus=second_bus,
            source=ServiceRef(name="orion-cortex-exec"),
            snapshot=snapshot,
            concepts=concepts,
            correlation_id="corr-concept-2",
        )
    )

    assert first.status == "written"
    assert first.graph == "orion:self:induced"
    assert second.status == "skipped"
    assert second.detail == "unchanged_concepts"
    assert [channel for channel, _ in first_bus.published] == ["orion:rdf:enqueue"]
    assert second_bus.published == []


def test_self_concept_induce_verb_publishes_induced_graph_only():
    self_study._LAST_CONCEPT_PUBLISH_KEY = None
    bus = _FakeBus()
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": "corr-induce"})

    out, effects = asyncio.run(verb_adapters.SelfConceptInduceVerb().execute(ctx, _request("self_concept_induce")))

    assert effects == []
    data = json.loads(out.final_text)
    result = data["result"] if "result" in data else data
    assert result["graph_write"]["status"] == "written"
    assert result["graph_write"]["authoritative"] is False
    assert result["graph_write"]["graph"] == "orion:self:induced"
    assert result["summary"].startswith("Concept induction produced ")
    assert all(concept["trust_tier"] == "induced" for concept in result["concepts"])
    assert [channel for channel, _ in bus.published] == ["orion:rdf:enqueue"]


def test_reflect_self_concepts_returns_reflective_findings_with_evidence_and_concept_refs():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)

    findings = self_study.reflect_self_concepts(snapshot, concepts)

    assert findings
    assert all(finding.trust_tier == "reflective" for finding in findings)
    assert all(finding.evidence for finding in findings)
    assert all(finding.concept_refs for finding in findings)
    assert all(ref.trust_tier == "authoritative" for finding in findings for ref in finding.evidence)
    assert all(ref.trust_tier == "induced" for finding in findings for ref in finding.concept_refs)


def test_reflective_rdf_request_uses_separate_graph_and_never_authoritative():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)
    findings = self_study.reflect_self_concepts(snapshot, concepts)

    req = self_study.build_self_reflection_rdf_request(
        source_snapshot=snapshot,
        findings=findings,
        run_id="reflect-run-1",
    )

    assert req.graph == "orion:self:reflective"
    assert req.kind == "self.reflections.reflective.v1"
    assert req.payload["trust_tier"] == "reflective"
    assert "ReflectiveSelfFinding" in (req.triples or "")
    assert "AuthoritativeSelfFact" not in (req.triples or "")


def test_reflective_journal_payload_is_compact_and_marked_reflective():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)
    findings = self_study.reflect_self_concepts(snapshot, concepts)

    journal = self_study.build_self_reflection_journal_entry(
        snapshot=snapshot,
        findings=findings,
        correlation_id="corr-reflect-journal",
    )

    assert journal.source_kind == "self_reflection"
    assert journal.title == "Self-study reflective findings"
    assert "Trust tier: reflective." in journal.body
    assert journal.body.count("\n") < 10


def test_publish_reflection_gracefully_skips_without_bus():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)
    findings = self_study.reflect_self_concepts(snapshot, concepts)

    graph_status, journal_status, journal_entry = asyncio.run(
        self_study.publish_self_reflection_artifacts(
            bus=None,
            source=ServiceRef(name="orion-cortex-exec"),
            snapshot=snapshot,
            findings=findings,
            correlation_id="corr-reflect-1",
        )
    )

    assert graph_status.status == "skipped"
    assert graph_status.graph == "orion:self:reflective"
    assert journal_status.status == "skipped"
    assert journal_status.append_only is True
    assert journal_entry.source_kind == "self_reflection"


def test_self_concept_reflect_verb_writes_reflective_graph_and_journal():
    self_study._LAST_REFLECTION_PUBLISH_KEY = None
    bus = _FakeBus()
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": "corr-reflect-2"})

    out, effects = asyncio.run(verb_adapters.SelfConceptReflectVerb().execute(ctx, _request("self_concept_reflect")))

    assert effects == []
    data = json.loads(out.final_text)
    result = data["result"] if "result" in data else data
    assert result["validated_phase2a"] is True
    assert result["graph_write"]["status"] == "written"
    assert result["graph_write"]["authoritative"] is False
    assert result["graph_write"]["graph"] == "orion:self:reflective"
    assert result["journal_write"]["status"] == "written"
    assert result["journal_entry"]["source_kind"] == "self_reflection"
    assert all(finding["trust_tier"] == "reflective" for finding in result["findings"])
    assert [channel for channel, _ in bus.published] == ["orion:rdf:enqueue", "orion:journal:write"]


def test_self_retrieve_factual_mode_returns_authoritative_records_only():
    request = self_study.SelfStudyRetrieveRequestV1.model_validate({"retrieval_mode": "factual"})

    result = self_study.retrieve_self_study(request)

    assert result.retrieval_mode == "factual"
    assert result.counts.total > 0
    assert result.counts.induced == 0
    assert result.counts.reflective == 0
    assert result.counts.facts == result.counts.total
    assert all(group.trust_tier == "authoritative" for group in result.groups)
    assert all(item.record_type == "fact" for group in result.groups for item in group.items)
    assert all(item.source_kind != "self_concept_reflect" for group in result.groups for item in group.items)


def test_self_retrieve_conceptual_mode_preserves_authoritative_and_induced_distinction():
    request = self_study.SelfStudyRetrieveRequestV1.model_validate(
        {
            "retrieval_mode": "conceptual",
            "filters": {"text_query": "recall", "limit": 20},
        }
    )

    result = self_study.retrieve_self_study(request)

    trust_tiers = {item.trust_tier for group in result.groups for item in group.items}
    assert result.retrieval_mode == "conceptual"
    assert "reflective" not in trust_tiers
    assert "authoritative" in trust_tiers
    assert "induced" in trust_tiers
    assert all(item.record_type in {"fact", "concept"} for group in result.groups for item in group.items)


def test_self_retrieve_reflective_mode_returns_all_three_trust_tiers():
    request = self_study.SelfStudyRetrieveRequestV1.model_validate(
        {
            "retrieval_mode": "reflective",
            "filters": {"limit": 30},
        }
    )

    result = self_study.retrieve_self_study(request)

    trust_tiers = {item.trust_tier for group in result.groups for item in group.items}
    assert trust_tiers == {"authoritative", "induced", "reflective"}
    assert result.counts.reflections > 0
    assert any(item.record_type == "reflection" for group in result.groups for item in group.items)


def test_self_retrieve_filters_by_kind_and_preserves_provenance_fields():
    request = self_study.SelfStudyRetrieveRequestV1.model_validate(
        {
            "retrieval_mode": "reflective",
            "filters": {
                "reflection_kinds": ["seam_risk"],
                "source_kinds": ["self_concept_reflect"],
                "limit": 10,
            },
        }
    )

    result = self_study.retrieve_self_study(request)

    assert result.counts.total > 0
    for group in result.groups:
        for item in group.items:
            assert item.reflection_kind == "seam_risk"
            assert item.source_kind == "self_concept_reflect"
            assert item.source_snapshot_id.startswith("self-snapshot-")
            assert item.trust_tier == "reflective"


def test_self_retrieve_never_upcasts_and_reports_unqueried_external_surfaces():
    request = self_study.SelfStudyRetrieveRequestV1.model_validate(
        {
            "retrieval_mode": "reflective",
            "filters": {"storage_surfaces": ["in_process"], "limit": 20},
        }
    )

    result = self_study.retrieve_self_study(request)

    assert all(
        not (item.record_type in {"concept", "reflection"} and item.trust_tier == "authoritative")
        for group in result.groups
        for item in group.items
    )
    statuses = {status.storage_surface: status.status for status in result.backend_status}
    assert statuses["in_process"] == "used"
    assert statuses["rdf_graph"] == "not_queried"
    assert statuses["journal"] == "not_queried"
    assert any("does not widen self.factual.v1" in note for note in result.notes)


def test_self_retrieve_verb_returns_typed_result():
    bus = _FakeBus()
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": "corr-retrieve"})

    out, effects = asyncio.run(
        verb_adapters.SelfRetrieveVerb().execute(
            ctx,
            _request(
                "self_retrieve",
                extra={
                    "retrieval_mode": "conceptual",
                    "filters": {"text_query": "runtime", "limit": 8},
                },
            ),
        )
    )

    assert effects == []
    data = json.loads(out.final_text)
    result = data["result"] if "result" in data else data
    assert result["retrieval_mode"] == "conceptual"
    assert result["counts"]["total"] > 0
    assert result["counts"]["reflective"] == 0
    assert all(group["trust_tier"] in {"authoritative", "induced"} for group in result["groups"])
    assert bus.published == []


def test_publish_self_study_artifacts_gracefully_skips_without_bus():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    graph_status, journal_status, journal_entry = asyncio.run(
        self_study.publish_self_study_artifacts(
            bus=None,
            source=ServiceRef(name="orion-cortex-exec"),
            snapshot=snapshot,
            correlation_id="corr-2",
        )
    )

    assert graph_status.status == "skipped"
    assert graph_status.authoritative is True
    assert journal_status.status == "skipped"
    assert journal_status.authoritative is False
    assert journal_status.append_only is True
    assert journal_entry.source_kind == "self_study"


def test_self_repo_inspect_verb_publishes_graph_and_journal():
    bus = _FakeBus()
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": "corr-3"})

    out, effects = asyncio.run(verb_adapters.SelfRepoInspectVerb().execute(ctx, _request()))

    assert effects == []
    data = json.loads(out.final_text)
    assert data["snapshot"]["trust_tier"] == "authoritative"
    assert data["graph_write"]["status"] == "written"
    assert data["graph_write"]["authoritative"] is True
    assert data["journal_write"]["status"] == "written"
    assert data["journal_write"]["authoritative"] is False
    assert [channel for channel, _ in bus.published] == ["orion:rdf:enqueue", "orion:journal:write"]
    assert bus.published[0][1].kind == "rdf.write.request"
    assert bus.published[1][1].kind == "journal.entry.write.v1"


def test_self_repo_inspect_reports_partial_backend_failure():
    self_study._LAST_GRAPH_PUBLISH_KEY = None
    bus = _FakeBus(fail_channel="orion:journal:write")
    result = asyncio.run(
        self_study.run_self_repo_inspect(
            bus=bus,
            source=ServiceRef(name="orion-cortex-exec"),
            correlation_id="corr-4",
        )
    )

    assert result.graph_write.status == "written"
    assert result.journal_write.status == "failed"
    assert result.journal_write.authoritative is False
    assert "publish_failed:orion:journal:write" in (result.journal_write.detail or "")


def test_repeat_publish_skips_unchanged_graph_but_keeps_journal_append_intent():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    self_study._LAST_GRAPH_PUBLISH_KEY = None

    first_bus = _FakeBus()
    first_graph, first_journal, _ = asyncio.run(
        self_study.publish_self_study_artifacts(
            bus=first_bus,
            source=ServiceRef(name="orion-cortex-exec"),
            snapshot=snapshot,
            correlation_id="corr-5",
        )
    )
    second_bus = _FakeBus()
    second_graph, second_journal, _ = asyncio.run(
        self_study.publish_self_study_artifacts(
            bus=second_bus,
            source=ServiceRef(name="orion-cortex-exec"),
            snapshot=snapshot,
            correlation_id="corr-6",
        )
    )

    assert first_graph.status == "written"
    assert second_graph.status == "skipped"
    assert second_graph.detail == "unchanged_snapshot"
    assert second_journal.status == "written"
    assert second_journal.append_only is True
    assert [channel for channel, _ in first_bus.published] == ["orion:rdf:enqueue", "orion:journal:write"]
    assert [channel for channel, _ in second_bus.published] == ["orion:journal:write"]
