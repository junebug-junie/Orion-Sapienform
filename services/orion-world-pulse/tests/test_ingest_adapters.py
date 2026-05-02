from __future__ import annotations

from pathlib import Path

from app.services.ingest import html_section_adapter, manual_urls_adapter, rss_adapter, sitemap_adapter
from orion.schemas.world_pulse import WorldPulseSourceV1

FIXTURES = Path(__file__).parent / "fixtures"


def _source(strategy: str) -> WorldPulseSourceV1:
    return WorldPulseSourceV1(
        source_id=f"source-{strategy}",
        name=f"Source {strategy}",
        type="rss",
        strategy=strategy,  # type: ignore[arg-type]
        url="https://example.org/feed.xml",
        domains=["example.org"],
        allowed_path_prefixes=["/news/path", "/news"],
        trust_tier=1,
        approved=True,
        enabled=True,
    )


def test_rss_adapter_parses_fixture(monkeypatch):
    xml = (FIXTURES / "sample_rss.xml").read_text(encoding="utf-8")
    source = _source("rss")
    monkeypatch.setattr(rss_adapter, "http_get_text", lambda *args, **kwargs: xml)
    rows = rss_adapter.fetch(source, 5)
    assert len(rows) == 2
    assert rows[0].url.startswith("https://example.org/news/")


def test_sitemap_adapter_bounds_child_sitemaps(monkeypatch):
    source = _source("sitemap").model_copy(
        update={"sitemap_max_child_sitemaps": 1, "sitemap_max_urls": 5, "allowed_path_prefixes": ["/sitemap", "/news"]}
    )
    index_xml = (FIXTURES / "sample_sitemap_index.xml").read_text(encoding="utf-8")
    child_xml = (FIXTURES / "sample_sitemap_child.xml").read_text(encoding="utf-8")

    calls = {"count": 0}

    def _http_get(url: str, **kwargs):  # noqa: ANN001
        if url.endswith("feed.xml"):
            return index_xml
        calls["count"] += 1
        return child_xml

    monkeypatch.setattr(sitemap_adapter, "http_get_text", _http_get)
    rows = sitemap_adapter.fetch(source, 5)
    assert calls["count"] == 1
    assert len(rows) == 2
    assert all("/news/" in row.url for row in rows)


def test_html_adapter_filters_to_allowed_domains_and_paths(monkeypatch):
    source = _source("html_section").model_copy(update={"url": "https://example.org/news/path/"})
    html = (FIXTURES / "sample_section.html").read_text(encoding="utf-8")
    monkeypatch.setattr(html_section_adapter, "http_get_text", lambda *args, **kwargs: html)
    rows = html_section_adapter.fetch(source, 5)
    assert len(rows) == 2
    assert all(candidate.url.startswith("https://example.org/news/path/") for candidate in rows)


def test_manual_adapter_enforces_domains():
    source = _source("manual_urls").model_copy(
        update={
            "urls": [
                "https://example.org/news/path/allowed",
                "https://unapproved.example.net/blocked",
            ],
            "max_articles_per_day": 10,
        }
    )
    rows = manual_urls_adapter.fetch(source, 5)
    assert len(rows) == 1
    assert rows[0].url == "https://example.org/news/path/allowed"

