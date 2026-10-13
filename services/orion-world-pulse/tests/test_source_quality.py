"""Source quality: article URLs only, freshness window, stuck-source detector.

Each case is pinned to a live failure from world_pulse_read_seed 2026-10-01..13
(see app/services/source_health.py's module docstring).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest

import app.services.curiosity as curiosity
from app.services import pipeline
from app.services.digest import curate_digest_items
from app.services.ingest import manual_urls_adapter, rss_adapter
from app.services.ingest.base import parse_date, to_candidate
from app.services.source_health import (
    assess_source_health,
    filter_fresh,
    load_previous_source_health,
    stuck_reason,
)
from app.services.source_registry import load_source_registry
from orion.world_pulse_read.reader_blocks import reader_block_reason
from orion.schemas.world_pulse import (
    DailyWorldPulseItemV1,
    SectionCoverageV1,
    SourceRegistryV1,
    WorldPulseAllowedUsesV1,
    WorldPulseSourceV1,
)

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _src(**over) -> WorldPulseSourceV1:
    base = dict(
        source_id="s1",
        name="S1",
        type="rss",
        strategy="rss",
        url="https://example.org/feed.xml",
        domains=["example.org"],
        trust_tier=1,
        approved=True,
        enabled=True,
    )
    base.update(over)
    return WorldPulseSourceV1(**base)


# --- dates -----------------------------------------------------------------


def test_parse_date_rfc822_iso_and_z():
    assert parse_date("Tue, 13 Oct 2026 17:56:08 +0000") == datetime(2026, 10, 13, 17, 56, 8, tzinfo=timezone.utc)
    # WHO's feed writes "Z" as the RFC 822 zone; Atom writes ISO 8601.
    assert parse_date("Wed, 25 Feb 2026 18:06:55 Z").year == 2026
    assert parse_date("2026-10-13T15:19:16-04:00") == datetime(2026, 10, 13, 19, 19, 16, tzinfo=timezone.utc)
    assert parse_date("2026-10-13T12:00:00Z").tzinfo is not None
    assert parse_date("2026-10-13").tzinfo is not None  # naive -> UTC
    assert parse_date("not a date") is None
    assert parse_date("") is None


# --- article URLs only -------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.usgs.gov/news/national-news-release",
        "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "https://example.org/",
        "https://example.org/news/",
    ],
)
def test_to_candidate_rejects_listing_and_data_urls(url):
    host = urlparse(url).hostname
    src = _src(domains=[host])
    assert to_candidate(source=src, url=url, title="t", summary=None, published_at=NOW, discovered_via="rss") is None


@pytest.mark.parametrize(
    "url",
    [
        # One real article per newly added source (sources.yaml), from the live feeds 2026-10-13.
        "https://www.pbs.org/newshour/politics/what-to-know-about-the-search-of-democratic-rep-omars-home",
        "https://rollcall.com/2026/10/13/house-republicans-open-seats-problem-midterm-elections/",
        "https://news.un.org/feed/view/en/story/2026/10/1168583",
        "https://www.deseret.com/politics/2026/10/12/john-curtis-mike-lee-opposite-ends-ukraine-russia-discourse/",
        "https://kffhealthnews.org/elections/online-safety-children-kosa-legislation-unites-democrats-republicans-big-tech/",
        "https://www.statnews.com/2026/10/13/helen-branswell-stat-reporter-obituary/?utm_campaign=rss",
        "https://www.sciencedaily.com/releases/2026/10/261012185023.htm",
        "https://therecord.media/sandbox-vulnerability-gives-rogue-ai-agents-path-out",
        "https://www.bleepingcomputer.com/news/microsoft/microsoft-windows-11-24h2-home-and-pro-reached-end-of-support/",
        "https://nvidianews.nvidia.com/news/nvidia-and-cisco-introduce-new-pluggable-optics-form-factor-to-scale-ai-factories",
        "https://blogs.nvidia.com/blog/ocp-global-summit-vera-rubin-scale-ai-factories/",
        "https://www.servethehome.com/intels-crescent-island-ai-server-gpu-gets-a-name-and-specs-meet-the-intel-gpu-d-series/",
        # Org-owned Hugging Face posts (182 of 876 feed items were rejected before the fix).
        "https://huggingface.co/blog/microsoft/thinkingbox",
        "https://huggingface.co/blog/ServiceNow-AI/autosynthdata",
    ],
)
def test_real_article_urls_from_new_sources_pass_their_source_rules(url):
    registry = load_source_registry(str(REPO_ROOT / "config" / "world_pulse" / "sources.yaml"))
    host = urlparse(url).hostname
    owners = [s for s in registry.sources if s.enabled and host in s.domains]
    assert owners, f"no enabled source owns {host}"
    assert any(
        to_candidate(source=s, url=url, title="t", summary=None, published_at=NOW, discovered_via="rss") for s in owners
    ), url


def test_manual_urls_index_page_yields_nothing():
    src = _src(
        strategy="manual_urls",
        type="manual",
        domains=["www.usgs.gov"],
        urls=["https://www.usgs.gov/news/national-news-release", "https://www.usgs.gov/news/featured-story"],
    )
    rows = manual_urls_adapter.fetch(src, 5)
    assert [r.url for r in rows] == ["https://www.usgs.gov/news/featured-story"]
    # ...and it is undated, so the freshness window drops it anyway.
    kept, dropped = filter_fresh(rows, now=NOW, max_age_days=3)
    assert kept == [] and dropped["undated"] == 1


_ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Article one</title>
    <link rel="self" href="https://example.org/feed/entry-1.xml"/>
    <link rel="alternate" href="https://example.org/2026/10/article-one"/>
    <published>2026-10-13T08:00:00Z</published>
  </entry>
  <entry>
    <title>Article one again</title>
    <link href="https://example.org/2026/10/article-one"/>
    <updated>2026-10-13T09:00:00Z</updated>
  </entry>
  <entry>
    <title>Section page</title>
    <link href="https://example.org/news/"/>
    <updated>2026-10-13T09:00:00Z</updated>
  </entry>
</feed>"""

_RSS_GUID = """<?xml version="1.0"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
  <item><title>Guid only</title><guid isPermaLink="true">https://example.org/2026/10/guid-only-story</guid>
    <dc:date>2026-10-12T10:00:00Z</dc:date></item>
  <item><title>Non-permalink guid</title><guid isPermaLink="false">abc-123</guid>
    <pubDate>Tue, 13 Oct 2026 10:00:00 +0000</pubDate></item>
</channel></rss>"""


def test_atom_prefers_alternate_link_dedupes_and_drops_listing(monkeypatch):
    monkeypatch.setattr(rss_adapter, "http_get_text", lambda *a, **k: _ATOM)
    rows = rss_adapter.fetch(_src(strategy="atom"), 5)
    assert [r.url for r in rows] == ["https://example.org/2026/10/article-one"]
    assert rows[0].published_at == datetime(2026, 10, 13, 8, 0, tzinfo=timezone.utc)


def test_rss_guid_permalink_fallback_and_dc_date(monkeypatch):
    monkeypatch.setattr(rss_adapter, "http_get_text", lambda *a, **k: _RSS_GUID)
    rows = rss_adapter.fetch(_src(), 5)
    assert [r.url for r in rows] == ["https://example.org/2026/10/guid-only-story"]
    assert rows[0].published_at is not None


# --- freshness window --------------------------------------------------------


class _C:
    def __init__(self, url, published):
        self.url = url
        self.published_at = published


def test_filter_fresh_drops_stale_and_undated_and_sorts_newest_first():
    items = [
        _C("old", NOW - timedelta(days=262)),  # the re-served 2026-01-24 WHO statement
        _C("mid", NOW - timedelta(days=2)),
        _C("none", None),
        _C("new", NOW - timedelta(hours=1)),
        _C("edge", NOW - timedelta(days=3, seconds=1)),
    ]
    kept, dropped = filter_fresh(items, now=NOW, max_age_days=3)
    assert [c.url for c in kept] == ["new", "mid"]
    assert dropped == {"stale": 2, "undated": 1, "future": 0}


def test_filter_fresh_disabled_keeps_everything():
    items = [_C("old", NOW - timedelta(days=400)), _C("none", None)]
    kept, dropped = filter_fresh(items, now=NOW, max_age_days=0)
    assert [c.url for c in kept] == ["old", "none"]
    assert dropped == {"stale": 0, "undated": 0, "future": 0}


def test_filter_fresh_reads_model_dump_only_candidates():
    class Dumped:
        def model_dump(self, mode="python"):
            return {"published_at": (NOW - timedelta(hours=2)).isoformat()}

    kept, _ = filter_fresh([Dumped()], now=NOW, max_age_days=3)
    assert len(kept) == 1


# --- stuck-source detector ---------------------------------------------------


def _chain(url_sets, k=3):
    prev = None
    out = []
    for urls in url_sets:
        prev = assess_source_health({"s": {"urls": urls, "fetched": len(urls)}}, prev, stuck_after=k)
        out.append(prev["s"])
    return out


def test_same_url_every_digest_goes_stuck_after_k():
    # usgs.gov: the same index URL, 13 days running.
    runs = _chain([["u"]] * 5)
    assert [r["consecutive_no_new"] for r in runs] == [0, 1, 2, 3, 4]
    assert [r["stuck"] for r in runs] == [False, False, False, True, True]
    assert stuck_reason(runs[3]) == "same_urls"


def test_zero_fresh_items_goes_stuck_and_recovers_on_a_new_url():
    runs = _chain([[], [], [], ["a"], ["a"]])
    assert [r["stuck"] for r in runs] == [False, False, True, False, False]
    assert stuck_reason(runs[2]) == "zero_fresh_items"
    assert runs[3]["new"] == 1 and runs[3]["consecutive_no_new"] == 0


def test_rotating_fresh_feed_never_stuck():
    runs = _chain([["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"]])
    assert not any(r["stuck"] for r in runs)


def test_stuck_disabled_when_k_is_zero():
    runs = _chain([["u"]] * 6, k=0)
    assert not any(r["stuck"] for r in runs)


def test_fetch_error_reason_and_malformed_previous_tolerated():
    health = assess_source_health(
        {"s": {"urls": [], "fetched": 0, "fetch_error": True}},
        {"s": {"urls": "garbage", "consecutive_no_new": "x"}},
        stuck_after=1,
    )
    assert health["s"]["stuck"] and stuck_reason(health["s"]) == "fetch_error"


def test_history_read_fails_open_with_a_status():
    assert load_previous_source_health("", run_id="r") == (None, "no_db_url")
    # Nothing listens on port 1: a down DB degrades to "error", never raises.
    assert load_previous_source_health("postgresql://u:p@127.0.0.1:1/x", run_id="r") == (None, "error")


def test_rotating_between_the_same_stale_urls_goes_stuck():
    # who.int: 26 items over 2 URLs. A, B, A, B must not read as "new" each time.
    runs = _chain([["A"], ["B"], ["A"], ["B"], ["A"]])
    assert [r["new"] for r in runs] == [1, 1, 0, 0, 0]
    assert [r["stuck"] for r in runs] == [False, False, False, False, True]


def test_an_empty_or_failed_run_does_not_reset_the_count():
    prev = None
    seq = [{"urls": ["u"]}, {"urls": [], "fetch_error": True}, {"urls": ["u"]}, {"urls": ["u"]}]
    out = []
    for row in seq:
        prev = assess_source_health({"s": row}, prev, stuck_after=3)
        out.append(prev["s"])
    assert [r["consecutive_no_new"] for r in out] == [0, 1, 2, 3]
    assert out[-1]["stuck"] is True
    assert out[1]["seen_urls"] == ["u"]


def test_seen_urls_are_capped():
    from app.services.source_health import MAX_SEEN_URLS

    prev = None
    for i in range(20):
        prev = assess_source_health({"s": {"urls": [f"u{i}-{j}" for j in range(10)]}}, prev, stuck_after=3)
    assert len(prev["s"]["seen_urls"]) == MAX_SEEN_URLS
    assert prev["s"]["seen_urls"][0] == "u19-0"


def test_filter_fresh_drops_far_future_dates():
    items = [_C("future", NOW + timedelta(days=30)), _C("soon", NOW + timedelta(hours=2))]
    kept, dropped = filter_fresh(items, now=NOW, max_age_days=3)
    assert [c.url for c in kept] == ["soon"]
    assert dropped["future"] == 1


def _registry():
    now = datetime.now(timezone.utc)
    return SourceRegistryV1(
        sources=[
            WorldPulseSourceV1(
                source_id="stuck-feed",
                name="Stuck",
                type="rss",
                strategy="rss",
                url="https://example.com/rss",
                domains=["example.com"],
                categories=["healthcare_mental_health"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
            WorldPulseSourceV1(
                source_id="fresh-feed",
                name="Fresh",
                type="rss",
                strategy="rss",
                url="https://example.net/rss",
                domains=["example.net"],
                categories=["us_politics"],
                trust_tier=1,
                enabled=True,
                approved=True,
                allowed_uses=WorldPulseAllowedUsesV1(),
                created_at=now,
            ),
        ]
    )


def test_digest_disabled_source_is_not_observed(monkeypatch):
    registry = _registry()
    registry.sources[0] = registry.sources[0].model_copy(
        update={"allowed_uses": WorldPulseAllowedUsesV1(digest=False)}
    )
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: registry)
    monkeypatch.setattr(
        _C,
        "model_dump",
        lambda self, mode="python": {"title": "t", "url": self.url, "summary": "s", "published_at": self.published_at},
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "fetch_source_candidates",
        lambda source, timeout_seconds: [_C(f"https://{source.domains[0]}/2026/10/a-story", now)],
    )
    monkeypatch.setattr(pipeline, "load_previous_source_health", lambda db_url, run_id: (None, "no_row"))
    result = pipeline.run_world_pulse(run_id="nodigest", requested_by="test", dry_run=True)
    assert "stuck-feed" not in result.run.metrics["source_health"]
    assert "fresh-feed" in result.run.metrics["source_health"]
    assert result.run.metrics["source_health_history"] == "no_row"


def test_pipeline_drops_stale_items_and_flags_stuck_source(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(pipeline, "load_source_registry", lambda path: _registry())

    def _fetch(source, timeout_seconds):  # noqa: ANN001
        if source.source_id == "stuck-feed":
            return [_C("https://example.com/2026/01/old-statement", now - timedelta(days=262))]
        return [_C(f"https://example.net/2026/10/story-{i}", now - timedelta(hours=i)) for i in range(3)]

    # _C has no model_dump; give it one the pipeline's normalize step can use.
    def _dump(self, mode="python"):
        return {"title": self.url.rsplit("/", 1)[-1], "url": self.url, "summary": "s", "published_at": self.published_at}

    monkeypatch.setattr(_C, "model_dump", _dump, raising=False)
    monkeypatch.setattr(pipeline, "fetch_source_candidates", _fetch)
    monkeypatch.setattr(
        pipeline,
        "load_previous_source_health",
        lambda db_url, run_id: (
            {
                "stuck-feed": {"urls": [], "consecutive_no_new": 2},
                "fresh-feed": {"urls": ["https://example.net/2026/10/story-0"], "consecutive_no_new": 0},
            },
            "loaded",
        ),
    )
    result = pipeline.run_world_pulse(run_id="stuck", requested_by="test", dry_run=True)
    health = result.run.metrics["source_health"]
    assert health["stuck-feed"]["stale_dropped"] == 1
    assert health["stuck-feed"]["fresh"] == 0
    assert health["stuck-feed"]["stuck"] is True
    assert health["fresh-feed"]["stuck"] is False and health["fresh-feed"]["new"] == 2
    assert "source_stuck:stuck-feed:zero_fresh_items:3" in result.run.warnings
    assert result.run.metrics["stuck_sources"] == ["stuck-feed"]
    assert result.run.metrics["source_health_history"] == "loaded"
    urls = {a["url"] for a in result.publish_status["articles"]}
    assert "https://example.com/2026/01/old-statement" not in urls
    assert len(urls) == 3


# --- digest breadth ----------------------------------------------------------


def _item(i, cat, conf):
    return DailyWorldPulseItemV1(
        item_id=f"item:{i}",
        run_id="r",
        title=f"t{i}",
        category=cat,
        summary="s",
        why_it_matters="w",
        what_changed="c",
        orion_read="r",
        confidence=conf,
        created_at=NOW,
    )


def test_curate_gives_every_section_one_card_before_any_gets_two():
    items = [_item(i, "us_politics", 0.9) for i in range(3)]
    items += [_item(10 + i, "security_infrastructure_software", 0.9) for i in range(3)]
    items += [_item(20, "healthcare_mental_health", 0.3)]
    chosen = curate_digest_items(
        items=items, required_sections=["us_politics"], max_total=3, max_per_section=2, min_per_required=1
    )
    assert sorted(i.category for i in chosen) == [
        "healthcare_mental_health",
        "security_infrastructure_software",
        "us_politics",
    ]


# --- curiosity gap-fill ------------------------------------------------------


def test_curiosity_drops_listing_page_findings(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setattr(curiosity, "_gate_open", lambda run_id, surprise_source=None: True)

    async def _backend(query, *, max_articles):
        arts = [
            {"url": "https://www.tomshardware.com/news", "title": "Tom's news", "description": "gpu"},
            {"url": "https://wccftech.com/topic/hardware/", "title": "hw topic", "description": "gpu"},
            {"url": "https://example.org/2026/10/real-gpu-story", "title": "GPU story", "description": "gpu"},
        ]
        return {"success": True, "urls": [a["url"] for a in arts], "articles": arts}

    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage={"hardware_compute_gpu": SectionCoverageV1(status="missing")},
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_backend,
        surprise_source=lambda: None,
    )
    assert [a.url for a in result[0].articles] == ["https://example.org/2026/10/real-gpu-story"]


# --- the real registry -------------------------------------------------------

def test_registry_has_no_enabled_source_the_reader_cannot_use():
    registry = load_source_registry(str(REPO_ROOT / "config" / "world_pulse" / "sources.yaml"))
    for s in registry.sources:
        if not (s.enabled and s.approved):
            continue
        hosts = set(s.domains) | {urlparse(u or "").hostname or "" for u in [s.url, *(s.urls or [])]}
        # Same list the reader skips on (PR #2634, live refusal evidence).
        assert not any(reader_block_reason(f"https://{h}/x") for h in hosts if h), s.source_id
        # Undated manual URLs can never pass the freshness window.
        assert s.strategy != "manual_urls", s.source_id
        assert s.url or s.urls, s.source_id


def test_every_digest_section_has_an_enabled_feed():
    registry = load_source_registry(str(REPO_ROOT / "config" / "world_pulse" / "sources.yaml"))
    covered = {c for s in registry.sources if s.enabled and s.approved for c in s.categories}
    for section in pipeline.SECTION_NAMES:
        assert section in covered, section


def test_arxiv_gets_a_wider_freshness_window_than_news():
    # arXiv dates papers by submission; a Monday run sees papers submitted Thu/Fri.
    # Under the 3-day news window those would all drop as stale.
    registry = load_source_registry(str(REPO_ROOT / "config" / "world_pulse" / "sources.yaml"))
    arxiv = next(s for s in registry.sources if s.source_id == "arxiv_research")
    assert arxiv.max_item_age_days is not None and arxiv.max_item_age_days >= 5
    news = [s for s in registry.sources if s.enabled and s.approved and s.source_id != "arxiv_research"]
    assert all(s.max_item_age_days is None for s in news)
