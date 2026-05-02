from __future__ import annotations

from datetime import datetime, timezone

from app.services import pipeline
from orion.schemas.world_pulse import SourceRegistryV1, WorldPulseAllowedUsesV1, WorldPulseSourceV1


def test_run_world_pulse_with_fixture_fetch(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "fetch_source_candidates",
        lambda source, timeout_seconds: [
            type("Candidate", (), {"model_dump": lambda self, mode="python": {
                "title": "US Senate passes AI infrastructure package",
                "url": "https://example.com/us-ai",
                "summary": "A policy update with GPU implications.",
                "author": "Staff",
                "published_at": None,
            }})(),
            type("Candidate", (), {"model_dump": lambda self, mode="python": {
                "title": "Utah county announces local election timeline",
                "url": "https://example.com/utah-election",
                "summary": "Local politics and administration changes.",
                "author": "Desk",
                "published_at": None,
            }})(),
        ],
    )
    result = pipeline.run_world_pulse(run_id="run-fixture", requested_by="test", dry_run=True)
    assert result.run.run_id == "run-fixture"
    assert result.run.status in {"completed", "partial"}
    if result.digest:
        assert result.digest.items


def test_pipeline_skips_unapproved_sources(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        sources=[
            WorldPulseSourceV1(
                source_id="unapproved",
                name="Unapproved",
                type="rss",
                url="https://example.com/rss",
                trust_tier=2,
                enabled=True,
                approved=False,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            )
        ]
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)
    monkeypatch.setattr(pipeline, "fetch_source_candidates", lambda source, timeout_seconds: [])
    result = pipeline.run_world_pulse(run_id="skip", requested_by="scheduler", dry_run=True)
    assert result.run.status == "failed"
    assert result.run.articles_accepted == 0
    assert result.run.sources_skipped == 1


def test_pipeline_fixture_mode_without_network(monkeypatch):
    calls = {"count": 0}

    def _boom(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("network should not be used")

    monkeypatch.setattr(pipeline, "fetch_source_candidates", _boom)
    fixtures = [
        {
            "title": "Global election update",
            "url": "https://example.com/g1",
            "summary": "summary",
            "published_at": datetime.now(timezone.utc),
        }
    ]
    result = pipeline.run_world_pulse(run_id="fixture", requested_by="test", dry_run=True, fixture_items=fixtures)
    assert result.run.status in {"completed", "partial"}
    assert calls["count"] == 0


def test_low_trust_digest_only_blocks_graph_and_stance(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        sources=[
            WorldPulseSourceV1(
                source_id="lowtrust",
                name="LowTrust",
                type="rss",
                url="https://example.com/rss",
                trust_tier=5,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(
                    digest=True,
                    claim_extraction=True,
                    graph_write=False,
                    stance_capsule=False,
                    prior_update_candidate=False,
                ),
                created_at=now,
            )
        ]
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)
    monkeypatch.setattr(pipeline, "fetch_source_candidates", lambda source, timeout_seconds: [])
    result = pipeline.run_world_pulse(run_id="lowtrust", requested_by="test", dry_run=True)
    assert result.run.status == "failed"


def test_pipeline_mixed_strategy_dispatch_and_error_isolation(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        sources=[
            WorldPulseSourceV1(
                source_id="rss-source",
                name="RSS",
                type="rss",
                strategy="rss",
                url="https://example.com/rss",
                domains=["example.com"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
            WorldPulseSourceV1(
                source_id="manual-source",
                name="Manual",
                type="manual",
                strategy="manual_urls",
                urls=["https://example.com/news/a"],
                domains=["example.com"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
            WorldPulseSourceV1(
                source_id="broken",
                name="Broken",
                type="rss",
                strategy="rss",
                url="https://example.com/broken",
                domains=["example.com"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
        ]
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def __init__(self, title: str, url: str) -> None:
            self.title = title
            self.url = url

        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": self.title, "url": self.url, "summary": "summary", "published_at": now}

    def _fetch(source, timeout_seconds):  # noqa: ANN001
        if source.source_id == "broken":
            raise RuntimeError("boom")
        if source.source_id == "manual-source":
            return [_Candidate("Manual A", "https://example.com/news/a")]
        return [_Candidate("RSS A", "https://example.com/news/b")]

    monkeypatch.setattr(pipeline, "fetch_source_candidates", _fetch)
    result = pipeline.run_world_pulse(run_id="mixed", requested_by="test", dry_run=True)
    assert result.run.status == "completed"
    assert result.run.sources_failed == 1
    assert "source_fetch_error:broken" in result.run.metrics.get("source_errors", [])
    assert "optional_source_fetch_error:broken" in result.run.warnings
    assert result.run.sources_fetched >= 1
    assert result.run.articles_accepted >= 1
    assert result.digest is not None


def test_required_source_failure_marks_partial(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        sources=[
            WorldPulseSourceV1(
                source_id="required-rss",
                name="Required RSS",
                type="rss",
                strategy="rss",
                url="https://example.com/rss",
                domains=["example.com"],
                trust_tier=1,
                enabled=True,
                approved=True,
                required=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
            WorldPulseSourceV1(
                source_id="manual-source",
                name="Manual",
                type="manual",
                strategy="manual_urls",
                urls=["https://example.com/news/a"],
                domains=["example.com"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
        ]
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def __init__(self, title: str, url: str) -> None:
            self.title = title
            self.url = url

        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": self.title, "url": self.url, "summary": "summary", "published_at": now}

    def _fetch(source, timeout_seconds):  # noqa: ANN001
        if source.source_id == "required-rss":
            raise RuntimeError("required source unavailable")
        return [_Candidate("Manual A", "https://example.com/news/a")]

    monkeypatch.setattr(pipeline, "fetch_source_candidates", _fetch)
    result = pipeline.run_world_pulse(run_id="required", requested_by="test", dry_run=True)
    assert result.run.status == "partial"
    assert "source_fetch_error:required-rss" in result.run.errors


def test_science_only_sources_report_sparse_coverage_with_completed_run(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        required_sections=["us_politics", "global_politics", "local_politics"],
        recommended_sections=["science_climate_energy", "ai_technology"],
        sources=[
            WorldPulseSourceV1(
                source_id="science-only",
                name="Science",
                type="rss",
                strategy="rss",
                url="https://example.com/science",
                domains=["example.com"],
                categories=["science_climate_energy"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": "Science A", "url": "https://example.com/news/sci-a", "summary": "summary", "published_at": now}

    monkeypatch.setattr(pipeline, "fetch_source_candidates", lambda source, timeout_seconds: [_Candidate()])
    result = pipeline.run_world_pulse(run_id="coverage-sparse", requested_by="test", dry_run=True)
    assert result.run.status == "completed"
    assert result.digest is not None
    assert result.digest.coverage_status == "sparse"
    assert "us_politics" in result.run.metrics["missing_required_sections"]
    assert result.digest.section_coverage["science_climate_energy"].status == "covered"


def test_required_sections_covered_reports_complete(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        required_sections=["us_politics", "global_politics", "local_politics"],
        recommended_sections=[],
        sources=[
            WorldPulseSourceV1(
                source_id="us",
                name="US",
                type="rss",
                strategy="rss",
                url="https://example.com/us",
                domains=["example.com"],
                categories=["us_politics"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
            WorldPulseSourceV1(
                source_id="global",
                name="Global",
                type="rss",
                strategy="rss",
                url="https://example.com/global",
                domains=["example.com"],
                categories=["global_politics"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
            WorldPulseSourceV1(
                source_id="local",
                name="Local",
                type="rss",
                strategy="rss",
                url="https://example.com/local",
                domains=["example.com"],
                categories=["local_politics"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
        ],
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def __init__(self, title: str, url: str):
            self.title = title
            self.url = url

        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": self.title, "url": self.url, "summary": "summary", "published_at": now}

    def _fetch(source, timeout_seconds):  # noqa: ANN001
        return [_Candidate(f"{source.source_id} title", f"https://example.com/news/{source.source_id}")]

    monkeypatch.setattr(pipeline, "fetch_source_candidates", _fetch)
    result = pipeline.run_world_pulse(run_id="coverage-complete", requested_by="test", dry_run=True)
    assert result.run.status == "completed"
    assert result.digest is not None
    assert result.digest.coverage_status == "complete"
    assert result.run.metrics["missing_required_sections"] == []


def test_disabled_source_is_skipped_not_failed(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        sources=[
            WorldPulseSourceV1(
                source_id="disabled-src",
                name="Disabled",
                type="rss",
                strategy="rss",
                url="https://example.com/disabled",
                domains=["example.com"],
                trust_tier=1,
                enabled=False,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)
    monkeypatch.setattr(pipeline, "fetch_source_candidates", lambda source, timeout_seconds: [])
    result = pipeline.run_world_pulse(run_id="disabled", requested_by="test", dry_run=True)
    assert result.run.sources_skipped == 1
    assert result.run.sources_failed == 0


def test_coverage_partial_when_required_covered_but_recommended_missing(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        required_sections=["us_politics"],
        recommended_sections=["ai_technology"],
        sources=[
            WorldPulseSourceV1(
                source_id="us",
                name="US",
                type="rss",
                strategy="rss",
                url="https://example.com/us",
                domains=["example.com"],
                categories=["us_politics"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": "US Politics", "url": "https://example.com/us-politics", "summary": "summary", "published_at": now}

    monkeypatch.setattr(pipeline, "fetch_source_candidates", lambda source, timeout_seconds: [_Candidate()])
    result = pipeline.run_world_pulse(run_id="coverage-partial", requested_by="test", dry_run=True)
    assert result.run.status == "completed"
    assert result.digest is not None
    assert result.digest.coverage_status == "partial"
    assert result.digest.section_coverage["us_politics"].status == "covered"
    assert "ai_technology" in result.run.metrics["missing_recommended_sections"]


def test_multi_category_source_counts_for_each_section(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        required_sections=[],
        recommended_sections=["science_climate_energy", "hardware_compute_gpu"],
        sources=[
            WorldPulseSourceV1(
                source_id="multi",
                name="Multi",
                type="rss",
                strategy="rss",
                url="https://example.com/multi",
                domains=["example.com"],
                categories=["science_climate_energy", "hardware_compute_gpu"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": "GPU and climate", "url": "https://example.com/multi-story", "summary": "summary", "published_at": now}

    monkeypatch.setattr(pipeline, "fetch_source_candidates", lambda source, timeout_seconds: [_Candidate()])
    result = pipeline.run_world_pulse(run_id="multi-category", requested_by="test", dry_run=True)
    assert result.digest is not None
    assert result.digest.section_coverage["science_climate_energy"].status == "covered"
    assert result.digest.section_coverage["hardware_compute_gpu"].status == "covered"


def test_cluster_consolidation_merges_same_topic_into_single_change(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        digest_policy={"max_digest_items_total": 12, "max_digest_items_per_section": 2, "min_digest_items_per_required_section": 1},
        situation_policy={"max_situation_changes_per_run": 25, "min_articles_for_corroborated_topic": 2},
        required_sections=["science_climate_energy"],
        sources=[
            WorldPulseSourceV1(
                source_id="science",
                name="Science",
                type="rss",
                strategy="rss",
                url="https://example.com/science",
                domains=["example.com"],
                categories=["science_climate_energy"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def __init__(self, title: str, url: str):
            self.title = title
            self.url = url

        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": self.title, "url": self.url, "summary": "summary", "published_at": now}

    monkeypatch.setattr(
        pipeline,
        "fetch_source_candidates",
        lambda source, timeout_seconds: [
            _Candidate("NASA mission update one", "https://example.com/news/1"),
            _Candidate("NASA mission update two", "https://example.com/news/2"),
            _Candidate("NASA mission update three", "https://example.com/news/3"),
        ],
    )
    result = pipeline.run_world_pulse(run_id="cluster-merge", requested_by="test", dry_run=True)
    assert result.run.status == "completed"
    assert result.run.articles_accepted == 3
    assert result.run.situation_changes_created <= 3
    assert result.digest is not None
    assert len(result.digest.items) <= 2


def test_digest_caps_limit_cards_but_preserve_article_evidence(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        digest_policy={"max_digest_items_total": 4, "max_digest_items_per_section": 2, "min_digest_items_per_required_section": 1},
        required_sections=["us_politics", "global_politics", "local_politics"],
        recommended_sections=["ai_technology"],
        sources=[
            WorldPulseSourceV1(
                source_id="mixed",
                name="Mixed",
                type="rss",
                strategy="rss",
                url="https://example.com/mixed",
                domains=["example.com"],
                categories=["us_politics", "global_politics", "local_politics", "ai_technology", "science_climate_energy"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def __init__(self, title: str, url: str):
            self.title = title
            self.url = url

        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": self.title, "url": self.url, "summary": "summary", "published_at": now}

    rows = [_Candidate(f"Story {i}", f"https://example.com/news/{i}") for i in range(20)]
    monkeypatch.setattr(pipeline, "fetch_source_candidates", lambda source, timeout_seconds: rows)
    result = pipeline.run_world_pulse(run_id="digest-cap", requested_by="test", dry_run=True)
    assert result.run.articles_accepted == 10
    assert result.digest is not None
    assert len(result.digest.items) <= 4
    assert result.digest.accepted_article_count == 10
    assert result.run.status == "completed"


def test_cluster_diagnostics_are_reported(monkeypatch):
    now = datetime.now(timezone.utc)
    registry = SourceRegistryV1(
        digest_policy={"max_digest_items_total": 6, "max_digest_items_per_section": 2, "min_digest_items_per_required_section": 1},
        clustering_policy={"enabled": True, "similarity_threshold": 0.35},
        sources=[
            WorldPulseSourceV1(
                source_id="source1",
                name="Source1",
                type="rss",
                strategy="rss",
                url="https://example.com/mixed",
                domains=["example.com"],
                categories=["science_climate_energy"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)

    class _Candidate:
        def __init__(self, title: str, url: str):
            self.title = title
            self.url = url

        def model_dump(self, mode: str = "python"):  # noqa: ANN001
            return {"title": self.title, "url": self.url, "summary": "summary", "published_at": now}

    rows = [
        _Candidate("Earthquake risk map update", "https://example.com/news/1"),
        _Candidate("Earthquake risk map analysis", "https://example.com/news/2"),
        _Candidate("GPU data center expansion", "https://example.com/news/3"),
        _Candidate("Wildfire readiness plan", "https://example.com/news/4"),
    ]
    monkeypatch.setattr(pipeline, "fetch_source_candidates", lambda source, timeout_seconds: rows)
    result = pipeline.run_world_pulse(run_id="diag", requested_by="test", dry_run=True)
    assert result.run.metrics["article_clusters"] < result.run.articles_accepted
    assert "singleton_cluster_count" in result.run.metrics
    assert "multi_article_cluster_count" in result.run.metrics
    assert "average_articles_per_cluster" in result.run.metrics
