from __future__ import annotations

from xml.etree import ElementTree

from app.services.ingest.base import http_get_text, parse_date, parse_xml, source_timeout_seconds, source_user_agent, to_candidate
from app.services.ingest.models import ArticleCandidate
from orion.schemas.world_pulse import WorldPulseSourceV1


def _read_feed_items(root: ElementTree.Element) -> list[dict[str, str | None]]:
    items: list[dict[str, str | None]] = []
    channel_items = root.findall(".//item")
    if channel_items:
        for item in channel_items:
            items.append(
                {
                    "title": item.findtext("title"),
                    "url": item.findtext("link"),
                    "summary": item.findtext("description"),
                    "published": item.findtext("pubDate"),
                }
            )
        return items
    for entry in root.findall(".//{*}entry"):
        link = ""
        link_node = entry.find("{*}link")
        if link_node is not None:
            link = link_node.attrib.get("href", "")
        items.append(
            {
                "title": entry.findtext("{*}title"),
                "url": link,
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
        if candidate:
            out.append(candidate)
    return out

