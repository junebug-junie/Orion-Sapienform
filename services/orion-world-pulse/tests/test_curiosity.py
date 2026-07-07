import app.services.curiosity as curiosity
from orion.schemas.world_pulse import SectionCoverageV1


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
