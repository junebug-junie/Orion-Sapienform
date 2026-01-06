from __future__ import annotations

from functools import lru_cache

from orion.spark.concept_induction.settings import ConceptSettings


@lru_cache(maxsize=1)
def get_settings() -> ConceptSettings:
    return ConceptSettings()


settings = get_settings()
