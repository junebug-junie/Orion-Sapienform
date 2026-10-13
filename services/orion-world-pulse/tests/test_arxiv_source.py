from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from app.services.cluster import build_article_clusters
from app.services.digest import curate_digest_items
from app.services.ingest import arxiv_api_adapter as ax
from app.services.ingest import registry
from app.services.normalize_article import normalize_article
from app.services.source_registry import load_source_registry
from orion.schemas.world_pulse import DailyWorldPulseItemV1, WorldPulseSourceV1

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCES_YAML = REPO_ROOT / "config" / "world_pulse" / "sources.yaml"


def _source(**update) -> WorldPulseSourceV1:
    base = WorldPulseSourceV1(
        source_id="arxiv_research",
        name="arXiv",
        type="api",
        strategy="api",
        urls=[
            "https://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate",
            "https://export.arxiv.org/api/query?search_query=cat:q-bio.NC&sortBy=submittedDate",
        ],
        domains=["arxiv.org", "export.arxiv.org"],
        allowed_path_prefixes=["/abs/"],
        categories=["ai_technology"],
        region_scope="technology",
        trust_tier=3,
        approved=True,
        enabled=True,
        max_articles_per_day=4,
    )
    return base.model_copy(update=update)


def _entry(arxiv_id: str) -> ax.ArxivEntry:
    return ax.ArxivEntry(arxiv_id=arxiv_id, title=f"Paper {arxiv_id}", summary="s", published_at=None, primary_category=None)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# --- id normalization -------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        # Live seed URLs from world_pulse_read_seed, 2026-10-13.
        ("https://arxiv.org/abs/2310.19279", "2310.19279"),
        ("https://arxiv.org/pdf/2509.07526.pdf", "2509.07526"),
        ("https://ar5iv.labs.arxiv.org/html/2609.10712", "2609.10712"),
        ("https://arxiv.org/abs/2609.10712v3", "2609.10712"),
        ("https://arxiv.org/pdf/2509.07526v2", "2509.07526"),
        ("http://export.arxiv.org/abs/2401.02385v1", "2401.02385"),
        ("https://www.arxiv.org/abs/2401.02385/", "2401.02385"),
        ("https://ar5iv.org/abs/2401.02385", "2401.02385"),
        ("http://arxiv.org/abs/hep-th/9901001v2", "hep-th/9901001"),
        ("https://arxiv.org/list/cs.AI/recent", None),
        ("https://arxiv.org/api/errors#incorrect_id", None),
        ("https://notarxiv.org/abs/2401.02385", None),
        ("https://example.com/abs/2401.02385", None),
        ("not a url", None),
    ],
)
def test_arxiv_id_from_url_normalizes_every_variant(url, expected):
    assert ax.arxiv_id_from_url(url) == expected


# --- Atom parsing -----------------------------------------------------------


def test_parse_arxiv_atom_fixture():
    entries = ax.parse_arxiv_atom((FIXTURES / "sample_arxiv_atom.xml").read_text(encoding="utf-8"))
    assert [e.arxiv_id for e in entries] == ["2610.13918", "2610.13419", "hep-th/9901001"]
    first = entries[0]
    # Title/summary newlines from the API are collapsed.
    assert first.title == "DyssomniaNN: Multi-Scale Spiking Neural Modeling of Human Sleep Disorders and Perceived Restedness"
    assert "\n" not in first.summary and first.summary.startswith("Clinical sleep medicine")
    assert first.primary_category == "cs.NE"
    assert first.categories == ["cs.NE", "cs.LG", "q-bio.NC"]
    assert first.authors == ["Daniel Szelogowski"]
    assert first.published_at == datetime(2026, 10, 9, 17, 52, 23, tzinfo=timezone.utc)
    # Version suffix stripped: the seed is the bare abs page.
    assert entries[1].abs_url == "https://arxiv.org/abs/2610.13419"


# --- selection: dedupe + cap ------------------------------------------------


def test_select_round_robins_across_categories():
    feeds = [[_entry("1.1"), _entry("1.2"), _entry("1.3")], [_entry("2.1"), _entry("2.2")], [_entry("3.1")]]
    picks = ax.select_papers(feeds, seen_ids=set(), cap=4)
    assert [p.arxiv_id for p in picks] == ["1.1", "2.1", "3.1", "1.2"]


def test_select_skips_seen_ids_and_cross_lists():
    feeds = [[_entry("1.1"), _entry("1.2")], [_entry("1.1"), _entry("2.1")]]
    picks = ax.select_papers(feeds, seen_ids={"1.2"}, cap=4)
    # 1.2 already read; 1.1 cross-listed into feed 2 is picked once.
    assert [p.arxiv_id for p in picks] == ["1.1", "2.1"]


def test_select_respects_cap_and_zero():
    feeds = [[_entry(f"1.{i}") for i in range(10)], [_entry(f"2.{i}") for i in range(10)]]
    assert len(ax.select_papers(feeds, seen_ids=set(), cap=4)) == 4
    assert ax.select_papers(feeds, seen_ids=set(), cap=0) == []


def test_seen_ids_from_live_seed_url_shapes():
    urls = [
        "https://arxiv.org/abs/2310.19279",
        "https://arxiv.org/pdf/2509.07526.pdf",
        "https://ar5iv.labs.arxiv.org/html/2609.10712",
        "https://www.coreweave.com/blog",
    ]
    assert ax.seen_ids_from_urls(urls) == {"2310.19279", "2509.07526", "2609.10712"}


# --- rate spacing -----------------------------------------------------------


def test_request_spacer_enforces_three_seconds():
    fake = _FakeClock()
    spacer = ax.RequestSpacer(clock=fake.clock, sleep=fake.sleep)
    spacer.wait()  # first request: no wait
    fake.now += 1.0
    spacer.wait()  # 1 s later: must wait the remaining 2 s
    spacer.wait()  # immediately again: full 3 s
    fake.now += 10.0
    spacer.wait()  # long gap: no wait
    assert fake.sleeps == pytest.approx([2.0, 3.0])


def test_spacing_floor_matches_arxiv_terms():
    assert ax.MIN_REQUEST_SPACING_SECONDS >= 3.0


# --- fetch end-to-end -------------------------------------------------------


def test_fetch_spaces_requests_identifies_orion_and_dedupes():
    xml = (FIXTURES / "sample_arxiv_atom.xml").read_text(encoding="utf-8")
    calls: list[tuple[str, str, float]] = []
    fake = _FakeClock()
    spacer = ax.RequestSpacer(clock=fake.clock, sleep=fake.sleep)

    def http_get(url, *, timeout_seconds, user_agent):
        calls.append((url, user_agent, fake.now))
        return xml

    rows = ax.fetch(
        _source(),
        20,
        seen_ids_loader=lambda: {"2610.13918"},
        http_get=http_get,
        spacer=spacer,
    )
    assert len(calls) == 2
    assert calls[1][2] - calls[0][2] >= 3.0
    ua = calls[0][1]
    assert "orion" in ua.lower() and "Mozilla" not in ua
    # 2610.13918 already read; both feeds return the same fixture, so the
    # remaining two papers are each picked once. hep-th/9901001's abs page
    # passes the /abs/ allowlist too.
    assert [r.metadata["arxiv_id"] for r in rows] == ["2610.13419", "hep-th/9901001"]
    first = rows[0]
    assert first.url == "https://arxiv.org/abs/2610.13419"
    assert first.discovered_via == "api"
    assert first.categories == ["ai_technology"]
    assert first.author == "A. Researcher, B. Researcher"


def test_fetch_survives_one_failed_category_but_raises_when_all_fail():
    xml = (FIXTURES / "sample_arxiv_atom.xml").read_text(encoding="utf-8")
    fake = _FakeClock()
    spacer = ax.RequestSpacer(clock=fake.clock, sleep=fake.sleep)

    def half(url, **_):
        if "cs.AI" in url:
            raise RuntimeError("503")
        return xml

    rows = ax.fetch(_source(), 20, seen_ids_loader=set, http_get=half, spacer=spacer)
    assert len(rows) == 3

    def boom(url, **_):
        raise RuntimeError("503")

    with pytest.raises(RuntimeError):
        ax.fetch(_source(), 20, seen_ids_loader=set, http_get=boom, spacer=spacer)


def test_fetch_without_history_still_returns_papers():
    xml = (FIXTURES / "sample_arxiv_atom.xml").read_text(encoding="utf-8")
    fake = _FakeClock()
    rows = ax.fetch(
        _source(max_articles_per_day=1),
        20,
        seen_ids_loader=lambda: None,
        http_get=lambda url, **_: xml,
        spacer=ax.RequestSpacer(clock=fake.clock, sleep=fake.sleep),
    )
    assert [r.metadata["arxiv_id"] for r in rows] == ["2610.13918"]


def test_registry_dispatches_arxiv_api_source(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(ax, "fetch", lambda source, timeout: seen.append(source.source_id) or [])
    registry.fetch_source_candidates(_source(), 20)
    assert seen == ["arxiv_research"]
    other = _source(source_id="other_api", urls=["https://api.example.org/q"])
    assert registry.fetch_source_candidates(other, 20) == []
    assert seen == ["arxiv_research"]


# --- config contract --------------------------------------------------------


def test_real_sources_yaml_ships_arxiv_on():
    reg = load_source_registry(str(SOURCES_YAML))
    src = next(s for s in reg.sources if s.source_id == "arxiv_research")
    assert src.enabled and src.approved and src.strategy == "api"
    assert src.categories == ["ai_technology"]  # an existing, consumed section
    assert 3 <= src.max_articles_per_day <= 5
    cats = [ax._query_category(u) for u in src.urls]
    assert cats == ["cs.AI", "cs.LG", "cs.CL", "q-bio.NC"]
    assert all("sortBy=submittedDate" in u and "sortOrder=descending" in u for u in src.urls)
    assert {urlsplit(u).hostname for u in src.urls} == {ax.ARXIV_API_HOST}
    assert reg.digest_policy["reserved_source_slots"]["arxiv_research"] == src.max_articles_per_day
    assert "arxiv_research" in reg.clustering_policy["singleton_source_ids"]


# --- clustering + digest: papers reach the digest ---------------------------


def _article(source: WorldPulseSourceV1, title: str, url: str):
    return normalize_article(
        run_id="r1",
        source=source,
        item={"title": title, "url": url, "summary": title},
        fetched_at=datetime(2026, 10, 13, tzinfo=timezone.utc),
    )


def test_arxiv_papers_never_merge_into_one_cluster():
    src = _source()
    arts = [
        _article(src, "Large Language Model Agents Reasoning Benchmark", "https://arxiv.org/abs/2610.00001"),
        _article(src, "Large Language Model Agents Reasoning Benchmark Revisited", "https://arxiv.org/abs/2610.00002"),
    ]
    merged = build_article_clusters(arts, clustering_policy={"similarity_threshold": 0.45})
    assert len(merged) == 1  # proves the two titles would merge by default
    split = build_article_clusters(
        arts, clustering_policy={"similarity_threshold": 0.45, "singleton_source_ids": ["arxiv_research"]}
    )
    assert len(split) == 2


def _item(item_id: str, category: str, source_id: str, confidence: float = 0.5) -> DailyWorldPulseItemV1:
    return DailyWorldPulseItemV1(
        item_id=item_id,
        run_id="r1",
        title=item_id,
        category=category,
        summary="s",
        why_it_matters="w",
        what_changed="c",
        orion_read="o",
        source_ids=[source_id],
        confidence=confidence,
        created_at=datetime(2026, 10, 13, tzinfo=timezone.utc),
    )


def test_reserved_slots_add_papers_without_displacing_news():
    news = [_item(f"n{i}", f"sec{i % 6}", "news", 0.9) for i in range(12)]
    hf = _item("hf", "ai_technology", "huggingface_blog", 0.6)
    papers = [_item(f"p{i}", "ai_technology", "arxiv_research", 0.99) for i in range(6)]
    kwargs = dict(required_sections=[], max_total=12, max_per_section=2, min_per_required=1)

    without = curate_digest_items(items=news + [hf], **kwargs)
    with_papers = curate_digest_items(items=news + [hf] + papers, reserved_source_slots={"arxiv_research": 4}, **kwargs)

    non_paper = [i.item_id for i in with_papers if not i.item_id.startswith("p")]
    assert non_paper == [i.item_id for i in without]  # nothing displaced, HF blog kept its slot
    assert [i.item_id for i in with_papers if i.item_id.startswith("p")] == ["p0", "p1", "p2", "p3"]


def test_without_reservation_papers_compete_as_before():
    papers = [_item(f"p{i}", "ai_technology", "arxiv_research", 0.99) for i in range(6)]
    out = curate_digest_items(items=papers, required_sections=[], max_total=12, max_per_section=2, min_per_required=1)
    assert len(out) == 2


# --- pipeline end-to-end: papers become reading seeds ----------------------


def test_pipeline_turns_arxiv_papers_into_reading_seeds(monkeypatch):
    """Real sources.yaml + real digest curation + Hub's real seed derivation:
    every picked paper becomes a digest_item seed on its abs page in
    ai_technology, and the news sections keep their cards."""
    from app.services import pipeline
    from app.services.ingest.models import ArticleCandidate
    from orion.world_pulse_read.seeds import seeds_from_digest_payload

    monkeypatch.setattr(pipeline.settings, "world_pulse_sources_config_path", str(SOURCES_YAML))
    monkeypatch.setattr(pipeline.settings, "world_pulse_fetch_enabled", True)
    monkeypatch.setattr(pipeline.settings, "world_pulse_curiosity_fetch_enabled", False)
    xml = (FIXTURES / "sample_arxiv_atom.xml").read_text(encoding="utf-8")
    fake = _FakeClock()
    words = ["harbor", "glacier", "orchid", "lantern", "meadow", "quartz", "falcon", "canyon"]

    def fetch(source, timeout_seconds):
        if source.source_id == "arxiv_research":
            return ax.fetch(
                source, timeout_seconds, seen_ids_loader=set, http_get=lambda url, **_: xml,
                spacer=ax.RequestSpacer(clock=fake.clock, sleep=fake.sleep),
            )
        host = (source.domains or ["example.org"])[0]
        return [
            ArticleCandidate(
                source_id=source.source_id, source_name=source.name,
                url=f"https://{host}/{source.source_id}/{w}", title=f"{source.source_id} {w} {i} report",
                fetched_at=datetime(2026, 10, 13, tzinfo=timezone.utc), discovered_via="rss",
                trust_tier=source.trust_tier, categories=list(source.categories),
            )
            for i, w in enumerate(words[:3])
        ]

    urls: dict[str, str] = {}
    real_normalize = pipeline.normalize_article

    def recording_normalize(**kw):
        art = real_normalize(**kw)
        urls[art.article_id] = art.url
        return art

    monkeypatch.setattr(pipeline, "fetch_source_candidates", fetch)
    monkeypatch.setattr(pipeline, "normalize_article", recording_normalize)
    result = pipeline.run_world_pulse(run_id="run-arxiv", requested_by="test", dry_run=True)
    assert result.digest is not None

    seeds = seeds_from_digest_payload(result.digest.model_dump(mode="json"), article_urls=urls)
    paper_seeds = [s for s in seeds if "arxiv.org" in s.url]
    assert sorted(s.url for s in paper_seeds) == [
        "https://arxiv.org/abs/2610.13419",
        "https://arxiv.org/abs/2610.13918",
        "https://arxiv.org/abs/hep-th/9901001",
    ]
    assert {s.section for s in paper_seeds} == {"ai_technology"}
    assert {s.kind for s in paper_seeds} == {"digest_item"}
    assert result.digest.max_digest_items_total == 16

    # Same run with arXiv returning nothing: the news cards are identical, so
    # the papers displaced nothing.
    def no_arxiv(source, timeout_seconds):
        return [] if source.source_id == "arxiv_research" else fetch(source, timeout_seconds)

    monkeypatch.setattr(pipeline, "fetch_source_candidates", no_arxiv)
    baseline = pipeline.run_world_pulse(run_id="run-arxiv", requested_by="test", dry_run=True)
    titles = lambda d: sorted(i.title for i in d.items if "arxiv_research" not in i.source_ids)  # noqa: E731
    assert titles(result.digest) == titles(baseline.digest)
    assert len(titles(baseline.digest)) >= 10
