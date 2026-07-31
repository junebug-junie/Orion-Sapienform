import pytest

import app.services.curiosity as curiosity
from orion.schemas.field_goal import FieldGoalProvenanceV1
from orion.schemas.world_pulse import SectionCoverageV1


def _goal() -> FieldGoalProvenanceV1:
    return FieldGoalProvenanceV1(
        artifact_id="goal-1",
        subject="attention",
        model_layer="field_attention",
        entity_id="node:substrate.biometrics",
        kind="memory.field_goals.proposed.v1",
        field_target_id="node:substrate.biometrics",
        target_kind="node",
        salience_score=0.8,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
        priority=0.8,
        proposal_status="proposed",
        provenance={"intake_channel": "internal.attention_runtime"},
    )


@pytest.fixture(autouse=True)
def _default_active_goal(monkeypatch):
    """SSP §6 Objective 6 (2026-07-30): _gate_open now reads
    orion.autonomy.goal_state.get_active_goal() instead of fabricating a synthetic
    goal per call. Default to a real, active goal (matching the old synthetic stub's
    always-present behavior) so existing tests don't need to know about this
    plumbing."""
    monkeypatch.setattr(curiosity, "get_active_goal", lambda: _goal())


def _coverage(**status_by_section) -> dict[str, SectionCoverageV1]:
    out = {}
    for section, status in status_by_section.items():
        out[section] = SectionCoverageV1(status=status)
    return out


async def _fake_backend(query, *, max_articles):
    return {
        "success": True,
        "urls": ["https://ex/1"],
        "articles": [
            {"url": "https://ex/1", "title": "GPU news", "description": "compute gpu"},
        ],
    }


def test_disabled_returns_empty():
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=False,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_dry_run_skips_fetch():
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=True,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_no_gaps_returns_empty():
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="covered"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_capability_denied_returns_empty(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "false")
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_fetches_under_covered_section(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(
            hardware_compute_gpu="missing", ai_technology="covered"
        ),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert len(result) == 1
    f = result[0]
    assert f.section == "hardware_compute_gpu"
    assert f.driving_gap == "missing"
    assert f.query == "hardware compute gpu recent news coverage"
    assert f.articles[0].url == "https://ex/1"
    # gap terms {hardware, compute, gpu}; article text "GPU news compute gpu"
    # overlaps {compute, gpu} => 2/3.
    assert round(f.articles[0].salience, 2) == 0.67


def test_surprise_source_reaches_gate_and_fetch_outcome(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    captured_notes = {}

    real_gate_open = curiosity._gate_open

    def _spy_gate_open(run_id, *, surprise_source=None):
        result = real_gate_open(run_id, surprise_source=surprise_source)
        ctx_score = curiosity._resolve_domain_surprise(surprise_source)
        captured_notes["score"] = ctx_score
        return result

    monkeypatch.setattr(curiosity, "_gate_open", _spy_gate_open)

    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing", ai_technology="covered"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
        surprise_source=lambda: 0.037,
    )
    assert len(result) == 1
    assert captured_notes["score"] == 0.037


def test_backend_error_degrades_to_no_followup(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")

    async def _boom(query, *, max_articles):
        raise RuntimeError("firecrawl down")

    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_boom,
    )
    # execute_readonly_fetch catches internally and returns a failed outcome
    # (success=False, no articles); we still record the followup with empty articles.
    assert len(result) == 1
    assert result[0].articles == []


def test_mapping_error_degrades_to_skip(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")

    class _BadArticle:
        url = "https://ex/1"
        title = "t"
        description = "d"
        salience = 5.0  # out of [0,1]; CuriosityFindingV1 validation will raise

    class _Outcome:
        action_id = "fetch-x"
        articles = [_BadArticle()]

    async def _fake_exec(req, *, fetch_backend):
        return _Outcome()

    monkeypatch.setattr(curiosity, "execute_readonly_fetch", _fake_exec)
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    # A schema-mapping failure must be caught per-section and never fail the run.
    assert result == []


def test_no_real_active_goal_denies_via_missing_goal(monkeypatch):
    """SSP §6 Objective 6 regression: with no real FieldGoalProvenanceV1 currently
    active (get_active_goal() returns None), the gate must deny via missing_goal,
    not silently fabricate one the way the old _synthetic_goal() stub did."""
    monkeypatch.setattr(curiosity, "get_active_goal", lambda: None)
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_gate_evaluation_error_degrades_to_empty(monkeypatch):
    def _boom_gate(run_id):
        raise FileNotFoundError("capability policy config missing")

    monkeypatch.setattr(curiosity, "_gate_open", _boom_gate)
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []
