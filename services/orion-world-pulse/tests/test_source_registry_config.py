from __future__ import annotations

from pathlib import Path

from app.services.source_registry import load_source_registry

# Repo root: services/orion-world-pulse/tests/.. /.. /..
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCES_YAML = REPO_ROOT / "config" / "world_pulse" / "sources.yaml"

# Regression guard: nasa_news was mistagged with hardware_compute_gpu even though
# NASA press releases (launches, contract awards, Mars missions) have nothing to
# do with GPU/hardware/compute tech. Because NASA publishes near-daily, that tag
# permanently marked hardware_compute_gpu as "covered" and silently suppressed
# the curiosity gap-fill (`build_curiosity_followups`) for that section for
# months. Do not re-add nasa_news, or any other off-topic source, to this
# category.
KNOWN_OFF_TOPIC_SOURCE_IDS_FOR_HARDWARE_COMPUTE_GPU = {"nasa_news"}
KNOWN_OFF_TOPIC_DOMAINS_FOR_HARDWARE_COMPUTE_GPU = {"nasa.gov", "www.nasa.gov"}


def _load_real_registry():
    return load_source_registry(str(SOURCES_YAML))


def test_nasa_news_is_not_tagged_hardware_compute_gpu():
    registry = _load_real_registry()
    nasa_sources = [s for s in registry.sources if s.source_id == "nasa_news"]
    assert nasa_sources, "expected nasa_news source to exist in sources.yaml"
    for source in nasa_sources:
        assert "hardware_compute_gpu" not in source.categories, (
            "nasa_news must not be tagged hardware_compute_gpu: NASA press "
            "releases are not GPU/hardware/compute content, and NASA's "
            "near-daily publication cadence permanently masks the curiosity "
            "gap-fill for this section (see docs/world_pulse_dev.md, "
            "'Known zero-source sections')."
        )


def test_no_known_off_topic_source_tagged_hardware_compute_gpu():
    """Thin regression guard, not a topic classifier.

    Asserts that none of the sources currently tagged hardware_compute_gpu
    match the known off-topic source_id/domain set that previously caused a
    months-long silent suppression of the curiosity gap-fill for this
    section. This intentionally does not attempt to judge topical relevance
    for arbitrary future sources.
    """
    registry = _load_real_registry()
    tagged = [s for s in registry.sources if "hardware_compute_gpu" in s.categories]
    for source in tagged:
        assert source.source_id not in KNOWN_OFF_TOPIC_SOURCE_IDS_FOR_HARDWARE_COMPUTE_GPU, (
            f"source {source.source_id!r} is a known off-topic source for "
            "hardware_compute_gpu and must not be re-tagged"
        )
        off_topic_domains = set(source.domains) & KNOWN_OFF_TOPIC_DOMAINS_FOR_HARDWARE_COMPUTE_GPU
        assert not off_topic_domains, (
            f"source {source.source_id!r} has known off-topic domain(s) "
            f"{off_topic_domains} for hardware_compute_gpu"
        )
