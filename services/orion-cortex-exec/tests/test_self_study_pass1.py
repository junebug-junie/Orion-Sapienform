import asyncio
import importlib.util
import json
import sys
import types
from collections import OrderedDict
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


def test_resolve_repo_root_prefers_explicit_env_override(monkeypatch, tmp_path):
    forced_root = tmp_path / "forced-root"
    forced_root.mkdir()
    monkeypatch.setenv("ORION_REPO_ROOT", str(forced_root))

    resolved = self_study._resolve_repo_root(tmp_path / "nested" / "app" / "self_study.py")

    assert resolved == forced_root.resolve()


def test_resolve_repo_root_detects_normal_repo_layout_from_markers(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    repo_root = tmp_path / "Orion-Sapienform"
    service_dir = repo_root / "services" / "orion-cortex-exec" / "app"
    service_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text('[tool.poetry]\nname = "orion"\n', encoding="utf-8")
    (repo_root / "orion").mkdir()
    (repo_root / "config").mkdir()

    resolved = self_study._resolve_repo_root(service_dir / "self_study.py")

    assert resolved == repo_root.resolve()


def test_resolve_repo_root_handles_flattened_container_layout_without_crashing(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    module_path = tmp_path / "app" / "app" / "self_study.py"

    resolved = self_study._resolve_repo_root(module_path)

    assert resolved == (tmp_path / "app").resolve()


def test_resolve_repo_root_fallback_is_deterministic_when_markers_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    module_path = tmp_path / "sandbox" / "nested" / "self_study.py"

    first = self_study._resolve_repo_root(module_path)
    second = self_study._resolve_repo_root(module_path)

    assert first == second == (tmp_path / "sandbox").resolve()


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
    # graph_surface concept retired 2026-07-28 along with orion-rdf-writer/
    # orion:rdf:enqueue -- there is no longer a graph write surface to induce.
    assert not any(concept.concept_kind == "graph_surface" for concept in concepts)

    summary = self_study.build_self_concept_summary(concepts)
    assert summary.startswith(f"Concept induction produced {len(concepts)} induced architectural concepts")


def test_graphify_derived_concepts_use_real_on_disk_graph_json():
    # No mock graph -- this repo's own graphify-out/graph.json, read the same
    # way the production code path reads it. Confirms the community mapping
    # actually overlaps real Layer-1 source paths and produces real,
    # evidence-backed concepts, not an empty-shell "ran but found nothing".
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    concepts = self_study._graphify_derived_concepts(snapshot, covered_item_ids=set())

    assert concepts, "expected at least one graphify_community concept from the real repo graph"
    assert all(concept.concept_kind == "graphify_community" for concept in concepts)
    assert all(concept.trust_tier == "induced" for concept in concepts)
    assert all(concept.evidence for concept in concepts)
    assert all(ref.trust_tier == "authoritative" for concept in concepts for ref in concept.evidence)
    # Idempotent: same snapshot + same on-disk graph -> identical concept ids.
    again = self_study._graphify_derived_concepts(snapshot, covered_item_ids=set())
    assert [c.concept_id for c in concepts] == [c.concept_id for c in again]


def test_graphify_derived_concepts_skip_already_covered_items():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    community_by_source_file = self_study._load_graphify_source_file_communities()
    assert community_by_source_file, "fixture depends on a non-empty graphify graph.json"

    all_item_ids = {
        item.item_id
        for section in ("services", "modules", "channels", "verbs", "schemas", "touchpoints", "env_surfaces")
        for item in getattr(snapshot, section)
    }

    concepts = self_study._graphify_derived_concepts(snapshot, covered_item_ids=all_item_ids)

    assert concepts == []


def _reset_structural_delta_state(monkeypatch):
    # snapshot_id is stable across build_self_snapshot() calls with the same
    # observed_at (confirmed by test_repeated_snapshot_keeps_stable_
    # snapshot_and_item_ids above), so every test in this file that uses the
    # standard fixture timestamp shares one cache key. Tests that care about
    # cold-start/idempotency semantics specifically must reset both the
    # "last observed" global AND the per-snapshot-id result cache, or an
    # earlier test's induce_self_concepts() call leaves a stale cached
    # result behind for later tests to (wrongly) hit.
    monkeypatch.setattr(self_study, "_LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA", None)
    monkeypatch.setattr(self_study, "_STRUCTURAL_DELTA_RESULT_CACHE", OrderedDict())
    # Explicit, not just "happens to be unset": a developer's shell exporting
    # this real env var would otherwise silently change these tests' cold-
    # start semantics (durable recovery would kick in unexpectedly).
    monkeypatch.setattr(self_study, "SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH", None)


def test_structural_delta_concepts_cold_start_produces_nothing(monkeypatch):
    _reset_structural_delta_state(monkeypatch)
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    first = self_study._structural_delta_concepts(snapshot)

    assert first == []
    # First observation is now recorded in-process -- confirms the cold-start
    # guard actually captured state rather than silently no-op'ing forever.
    assert self_study._LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA is not None


def test_structural_delta_concepts_unchanged_snapshot_stays_empty(monkeypatch):
    _reset_structural_delta_state(monkeypatch)
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    self_study._structural_delta_concepts(snapshot)  # cold start, primes state
    # Same snapshot_id -> memoized cache hit, NOT a fresh disk re-observation
    # (review fix): confirms repeat calls against one snapshot are pure.
    second = self_study._structural_delta_concepts(snapshot)

    assert second == []


def test_structural_delta_concepts_repeat_call_is_memoized_not_reobserved(monkeypatch):
    # Direct regression test for the review-caught crash: without
    # memoization, a second call against the same snapshot would treat its
    # own first call's "last seen" write as no-op and silently return a
    # different (empty) result than the first call, breaking Phase 2A's
    # idempotency invariant whenever a real delta exists on the first call.
    _reset_structural_delta_state(monkeypatch)
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    sentinel = [
        self_study._concept(
            snapshot=snapshot,
            concept_kind="structural_mass",
            label="repo-wide structural mass delta",
            description="sentinel delta for memoization test",
            evidence_items=[next(item for item in snapshot.modules if item.name == "orion.structural_mass")],
            inferred_from=["structural_mass_delta"],
        )
    ]
    self_study._STRUCTURAL_DELTA_RESULT_CACHE[snapshot.snapshot_id] = sentinel

    result = self_study._structural_delta_concepts(snapshot)

    assert result is sentinel


def test_structural_delta_concepts_persists_first_observation_to_durable_history(monkeypatch, tmp_path):
    # No prior anywhere (in-process or on disk) -- this is the true
    # cold-start baseline. It should still write ONE entry to the durable
    # log so a *future* process restart has something to recover.
    _reset_structural_delta_state(monkeypatch)
    history_path = tmp_path / "structural_mass_history.jsonl"
    monkeypatch.setattr(self_study, "SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH", str(history_path))
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    result = self_study._structural_delta_concepts(snapshot)

    assert result == []  # no prior to diff against -- still no fabricated concept
    history = self_study.read_snapshots(history_path)
    assert len(history) == 1
    assert history[0].commit_sha == self_study._LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA.commit_sha


def test_structural_delta_concepts_recovers_prior_from_durable_history_across_process_restart(monkeypatch, tmp_path):
    # Simulates a container restart: the in-process global is None (fresh
    # process) but an earlier process already recorded a real prior on the
    # durable volume. That recovered prior -- not "no prior at all" -- must
    # be what this call diffs against.
    _reset_structural_delta_state(monkeypatch)
    history_path = tmp_path / "structural_mass_history.jsonl"
    stale_prior = self_study.GraphSnapshotStats(
        node_count=1,
        edge_count=1,
        community_count=1,
        god_nodes=(),
        commit_sha="0" * 40,
        backfilled=True,
    )
    self_study.append_snapshot(history_path, stale_prior)
    monkeypatch.setattr(self_study, "SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH", str(history_path))
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    result = self_study._structural_delta_concepts(snapshot)

    # The real on-disk graph is certainly not identical to a 1-node/1-edge
    # stub at a fake all-zero SHA, so recovering `stale_prior` as the prior
    # must produce a real, non-trivial delta -- not silently stay empty the
    # way a true cold start would.
    assert len(result) == 1
    assert result[0].concept_kind == "structural_mass"
    # The real current commit differs from the seeded stale one, so this is
    # a genuinely new observation and gets appended alongside it.
    history = self_study.read_snapshots(history_path)
    assert len(history) == 2
    assert history[0].commit_sha == stale_prior.commit_sha


def test_structural_delta_concepts_does_not_reappend_unchanged_recovered_prior(monkeypatch, tmp_path):
    # First call establishes the real baseline and persists it.
    _reset_structural_delta_state(monkeypatch)
    history_path = tmp_path / "structural_mass_history.jsonl"
    monkeypatch.setattr(self_study, "SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH", str(history_path))
    first_snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    self_study._structural_delta_concepts(first_snapshot)
    assert len(self_study.read_snapshots(history_path)) == 1

    # Simulate a second process restart at the SAME commit: in-process state
    # resets to None again, but the durable log already has an entry for
    # this exact commit_sha. Recovering it must not re-append a duplicate.
    monkeypatch.setattr(self_study, "_LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA", None)
    monkeypatch.setattr(self_study, "_STRUCTURAL_DELTA_RESULT_CACHE", OrderedDict())
    second_snapshot = self_study.build_self_snapshot(observed_at="2026-03-22T00:00:00+00:00")

    result = self_study._structural_delta_concepts(second_snapshot)

    assert result == []  # same commit as the recovered prior -- no real delta
    assert len(self_study.read_snapshots(history_path)) == 1  # not duplicated


def test_semantic_enrichment_concepts_missing_cache_dir_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(self_study, "SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR", str(tmp_path / "not-mounted"))
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    concepts = self_study._semantic_enrichment_concepts(snapshot)

    assert concepts == []


def test_semantic_enrichment_concepts_reads_real_cache_entry_shape(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache" / "self_study_enrichment"
    entry_dir = cache_dir / "ab"
    entry_dir.mkdir(parents=True)
    (entry_dir / "abcdef123456.json").write_text(
        json.dumps(
            {
                "summary": "cortex-exec's self-study module builds a factual snapshot before any induction runs.",
                "touched_paths": ["services/orion-cortex-exec/app/self_study.py"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(self_study, "SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR", str(cache_dir))
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    concepts = self_study._semantic_enrichment_concepts(snapshot)

    assert len(concepts) == 1
    concept = concepts[0]
    assert concept.concept_kind == "semantic_enrichment"
    assert concept.description == "cortex-exec's self-study module builds a factual snapshot before any induction runs."
    assert concept.evidence
    assert concept.inferred_from == ["semantic_enrichment"]


def test_semantic_enrichment_concepts_skips_malformed_entries(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache" / "self_study_enrichment"
    entry_dir = cache_dir / "cd"
    entry_dir.mkdir(parents=True)
    (entry_dir / "no-summary.json").write_text(json.dumps({"touched_paths": ["services/orion-cortex-exec"]}), encoding="utf-8")
    (entry_dir / "not-json.json").write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(self_study, "SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR", str(cache_dir))
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    concepts = self_study._semantic_enrichment_concepts(snapshot)

    assert concepts == []


def test_reflect_self_concepts_does_not_crash_on_a_real_structural_delta(monkeypatch):
    # End-to-end regression test for the exact review-caught crash: prime a
    # fake-but-plausible "prior" GraphSnapshotStats (different from the real
    # on-disk graph.json) so the REAL _structural_delta_concepts() code path
    # computes a genuine non-trivial delta against the real current stats --
    # not a mocked function. Before the memoization fix, reflect_self_
    # concepts()'s internal validate_phase2a_induction() re-induction call
    # would silently drop that same concept on its second pass and raise
    # concept_idempotency_mismatch.
    _reset_structural_delta_state(monkeypatch)
    fake_prior = self_study.GraphSnapshotStats(
        node_count=1,
        edge_count=1,
        community_count=1,
        god_nodes=(),
        commit_sha="0000000000000000000000000000000000000000",
        backfilled=True,
    )
    monkeypatch.setattr(self_study, "_LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA", fake_prior)
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    concepts = self_study.induce_self_concepts(snapshot)
    assert any(c.concept_kind == "structural_mass" for c in concepts), (
        "fixture prior stats should differ enough from the real on-disk graph to produce a real delta"
    )

    # This must not raise concept_idempotency_mismatch.
    findings = self_study.reflect_self_concepts(snapshot, concepts)
    assert findings


def test_induce_self_concepts_still_idempotent_with_layer2_sources_wired(monkeypatch):
    # The core Phase 2A invariant (same snapshot in -> same concept ids out)
    # must survive the three additive Layer 2 sources above.
    _reset_structural_delta_state(monkeypatch)
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    first = self_study.induce_self_concepts(snapshot)
    second = self_study.induce_self_concepts(snapshot)

    assert [c.concept_id for c in first] == [c.concept_id for c in second]
    assert any(c.concept_kind == "graphify_community" for c in first)


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


def test_publish_self_concepts_is_permanently_retired():
    # orion:rdf:enqueue retired 2026-07-28: graph writeback is now a
    # permanent no-op regardless of whether the bus is connected or the
    # concepts changed between runs -- see self_study.py's
    # publish_self_concept_artifacts.
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)

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

    assert first.status == "skipped"
    assert first.detail == "channel_retired"
    assert first.graph == "orion:self:induced"
    assert second.status == "skipped"
    assert second.detail == "channel_retired"
    assert first_bus.published == []
    assert second_bus.published == []


def test_self_concept_induce_verb_graph_write_is_retired():
    bus = _FakeBus()
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": "corr-induce"})

    out, effects = asyncio.run(verb_adapters.SelfConceptInduceVerb().execute(ctx, _request("self_concept_induce")))

    assert effects == []
    data = json.loads(out.final_text)
    result = data["result"] if "result" in data else data
    assert result["graph_write"]["status"] == "skipped"
    assert result["graph_write"]["detail"] == "channel_retired"
    assert result["graph_write"]["authoritative"] is False
    assert result["graph_write"]["graph"] == "orion:self:induced"
    assert result["summary"].startswith("Concept induction produced ")
    assert all(concept["trust_tier"] == "induced" for concept in result["concepts"])
    assert bus.published == []


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


def test_reflect_self_concepts_pairs_structural_delta_with_enrichment_narrative(monkeypatch, tmp_path):
    # induce_self_concepts()'s Phase 2A validation re-runs induction and
    # requires byte-identical concept ids on repeat -- so instead of hand-
    # building a structural_mass concept that only the *first* call would
    # see, monkeypatch the two Layer 2 source functions themselves to be
    # deterministic. That way the real induce_self_concepts() pipeline (and
    # its internal repeat-call idempotency check) exercises the actual
    # pairing branch in reflect_self_concepts() end to end.
    cache_dir = tmp_path / "cache" / "self_study_enrichment"
    entry_dir = cache_dir / "ab"
    entry_dir.mkdir(parents=True)
    (entry_dir / "entry.json").write_text(
        json.dumps(
            {
                "summary": "structural_mass computes real git/graph deltas rather than fabricating a zero delta.",
                "touched_paths": ["orion/structural_mass/graph_delta.py"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(self_study, "SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR", str(cache_dir))

    def fake_structural_delta_concepts(snapshot):
        module_item = next(item for item in snapshot.modules if item.name == "orion.structural_mass")
        return [
            self_study._concept(
                snapshot=snapshot,
                concept_kind="structural_mass",
                label="repo-wide structural mass delta",
                description="graphify's on-disk structural snapshot moved: node_count_delta=+3.",
                evidence_items=[module_item],
                inferred_from=["structural_mass_delta"],
            )
        ]

    monkeypatch.setattr(self_study, "_structural_delta_concepts", fake_structural_delta_concepts)

    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)
    assert any(c.concept_kind == "structural_mass" for c in concepts)
    assert any(c.concept_kind == "semantic_enrichment" for c in concepts)

    findings = self_study.reflect_self_concepts(snapshot, concepts)

    paired = [f for f in findings if f.title == "Structural delta and a cached enrichment narrative both point at real change"]
    assert len(paired) == 1
    assert any(ref.concept_kind == "structural_mass" for ref in paired[0].concept_refs)
    assert any(ref.concept_kind == "semantic_enrichment" for ref in paired[0].concept_refs)


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


def test_self_concept_reflect_verb_graph_write_retired_journal_still_written():
    bus = _FakeBus()
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": "corr-reflect-2"})

    out, effects = asyncio.run(verb_adapters.SelfConceptReflectVerb().execute(ctx, _request("self_concept_reflect")))

    assert effects == []
    data = json.loads(out.final_text)
    result = data["result"] if "result" in data else data
    assert result["validated_phase2a"] is True
    assert result["graph_write"]["status"] == "skipped"
    assert result["graph_write"]["detail"] == "channel_retired"
    assert result["graph_write"]["authoritative"] is False
    assert result["graph_write"]["graph"] == "orion:self:reflective"
    assert result["journal_write"]["status"] == "written"
    assert result["journal_entry"]["source_kind"] == "self_reflection"
    assert all(finding["trust_tier"] == "reflective" for finding in result["findings"])
    assert [channel for channel, _ in bus.published] == ["orion:journal:write"]


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


def test_self_repo_inspect_verb_graph_write_retired_journal_still_written():
    bus = _FakeBus()
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="orion-cortex-exec"), "correlation_id": "corr-3"})

    out, effects = asyncio.run(verb_adapters.SelfRepoInspectVerb().execute(ctx, _request()))

    assert effects == []
    data = json.loads(out.final_text)
    assert data["snapshot"]["trust_tier"] == "authoritative"
    assert data["graph_write"]["status"] == "skipped"
    assert data["graph_write"]["detail"] == "channel_retired"
    assert data["graph_write"]["authoritative"] is True
    assert data["journal_write"]["status"] == "written"
    assert data["journal_write"]["authoritative"] is False
    assert [channel for channel, _ in bus.published] == ["orion:journal:write"]
    assert bus.published[0][1].kind == "journal.entry.write.v1"


def test_self_repo_inspect_reports_partial_backend_failure():
    bus = _FakeBus(fail_channel="orion:journal:write")
    result = asyncio.run(
        self_study.run_self_repo_inspect(
            bus=bus,
            source=ServiceRef(name="orion-cortex-exec"),
            correlation_id="corr-4",
        )
    )

    assert result.graph_write.status == "skipped"
    assert result.graph_write.detail == "channel_retired"
    assert result.journal_write.status == "failed"
    assert result.journal_write.authoritative is False
    assert "publish_failed:orion:journal:write" in (result.journal_write.detail or "")


def test_repeat_publish_graph_write_retired_but_keeps_journal_append_intent():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

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

    assert first_graph.status == "skipped"
    assert first_graph.detail == "channel_retired"
    assert second_graph.status == "skipped"
    assert second_graph.detail == "channel_retired"
    assert first_journal.status == "written"
    assert second_journal.status == "written"
    assert second_journal.append_only is True
    assert [channel for channel, _ in first_bus.published] == ["orion:journal:write"]
    assert [channel for channel, _ in second_bus.published] == ["orion:journal:write"]
