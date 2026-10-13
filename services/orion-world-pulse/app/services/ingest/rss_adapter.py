from __future__ import annotations

from xml.etree import ElementTree

from app.services.ingest.base import http_get_text, parse_date, parse_xml, source_timeout_seconds, source_user_agent, to_candidate
from app.services.ingest.models import ArticleCandidate
from orion.schemas.world_pulse import WorldPulseSourceV1

_DC_DATE = "{http://purl.org/dc/elements/1.1/}date"


def _rss_item_link(item: ElementTree.Element) -> str:
    """The item's own article URL: ``<link>``, else a permalink ``<guid>``."""
    link = (item.findtext("link") or "").strip()
    if link:
        return link
    guid = item.find("guid")
    if guid is not None and guid.attrib.get("isPermaLink", "true").lower() != "false":
        text = (guid.text or "").strip()
        if text.startswith(("http://", "https://")):
            return text
    return ""


def _atom_entry_link(entry: ElementTree.Element) -> str:
    """Prefer ``rel="alternate"`` (the article); skip self/enclosure/replies."""
    fallback = ""
    for node in entry.findall("{*}link"):
        href = (node.attrib.get("href") or "").strip()
        if not href:
            continue
        rel = node.attrib.get("rel", "alternate")
        if rel == "alternate":
            return href
        if not fallback and rel not in {"self", "enclosure", "replies", "edit"}:
            fallback = href
    return fallback


def _read_feed_items(root: ElementTree.Element) -> list[dict[str, str | None]]:
    items: list[dict[str, str | None]] = []
    # RSS 2.0 items are un-namespaced; RSS 1.0/RDF items are namespaced.
    channel_items = root.findall(".//item") or root.findall(".//{*}item")
    if channel_items:
        for item in channel_items:
            items.append(
                {
                    "title": item.findtext("title") or item.findtext("{*}title"),
                    "url": _rss_item_link(item) or (item.findtext("{*}link") or ""),
                    "summary": item.findtext("description") or item.findtext("{*}description"),
                    "published": item.findtext("pubDate") or item.findtext(_DC_DATE),
                }
            )
        return items
    for entry in root.findall(".//{*}entry"):
        items.append(
            {
                "title": entry.findtext("{*}title"),
                "url": _atom_entry_link(entry),
                "summary": entry.findtext("{*}summary") or entry.findtext("{*}content"),
                "published": entry.findtext("{*}published") or entry.findtext("{*}updated"),
            }
        )
    return items


def fetch(source: WorldPulseSourceV1, fallback_timeout_seconds: int) -> list[ArticleCandidate]:
    if not source.url:
        return []
    xml_text = http_get_text(
        source.url,
        timeout_seconds=source_timeout_seconds(source, fallback_timeout_seconds),
        user_agent=source_user_agent(source),
    )
    root = parse_xml(xml_text)
    out: list[ArticleCandidate] = []
    seen: set[str] = set()
    for row in _read_feed_items(root):
        candidate = to_candidate(
            source=source,
            url=(row.get("url") or "").strip(),
            title=(row.get("title") or "").strip(),
            summary=row.get("summary"),
            published_at=parse_date(row.get("published")),
            discovered_via="atom" if source.strategy == "atom" else "rss",
            metadata={"strategy": source.strategy or "rss"},
        )
        if candidate and candidate.url not in seen:
            seen.add(candidate.url)
            out.append(candidate)
    return out
