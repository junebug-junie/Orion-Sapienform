from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.world_pulse import RegionScope


class ArticleCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_name: str
    url: str
    canonical_url: str | None = None
    title: str
    subtitle: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    summary: str | None = None
    excerpt: str | None = None
    content_html: str | None = None
    raw_text: str | None = None
    fetched_at: datetime
    discovered_via: Literal["rss", "atom", "sitemap", "html_section", "manual_urls", "api"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    trust_tier: int = Field(ge=1, le=5)
    categories: list[str] = Field(default_factory=list)
    region_scope: RegionScope = "general_world"

