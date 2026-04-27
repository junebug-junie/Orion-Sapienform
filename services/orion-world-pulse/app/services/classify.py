from __future__ import annotations

from orion.schemas.world_pulse import ArticleRecordV1


_RULES: list[tuple[str, str]] = [
    ("utah", "local_politics"),
    ("senate", "us_politics"),
    ("congress", "us_politics"),
    ("election", "global_politics"),
    ("ai", "ai_technology"),
    ("gpu", "hardware_compute_gpu"),
    ("security", "security_infrastructure_software"),
    ("vulnerability", "security_infrastructure_software"),
    ("climate", "science_climate_energy"),
    ("energy", "science_climate_energy"),
    ("health", "healthcare_mental_health"),
]


def classify_article(article: ArticleRecordV1) -> str:
    text = f"{article.title} {article.text_excerpt or ''}".lower()
    for token, category in _RULES:
        if token in text:
            return category
    if article.categories:
        return article.categories[0]
    return "general_world"
