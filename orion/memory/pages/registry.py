"""Every core page the nightly fold can write, by slug. One place, read by orion-durable-runs
(``app/core_page_fold.py``, ``CORE_PAGE_FOLD_PAGES``) and ``scripts/core_page_dry_run.py``."""
from __future__ import annotations

from orion.memory.pages import juniper, lately, origin
from orion.memory.pages.fold import FoldSpec

FOLD_SPECS: dict[str, FoldSpec] = {spec.slug: spec for spec in (juniper.FOLD_SPEC, origin.FOLD_SPEC, lately.FOLD_SPEC)}


def selected_specs(pages) -> list[FoldSpec]:
    """``"juniper,origin"`` (or a list) -> the specs to fold, in that order, duplicates dropped.
    An unknown slug raises: a typo must not silently turn a page off."""
    slugs = pages.split(",") if isinstance(pages, str) else list(pages or [])
    out: list[FoldSpec] = []
    for raw in slugs:
        slug = str(raw).strip().lower()
        if not slug:
            continue
        if slug not in FOLD_SPECS:
            raise ValueError(f"unknown core page {slug!r}; known: {sorted(FOLD_SPECS)}")
        if FOLD_SPECS[slug] not in out:
            out.append(FOLD_SPECS[slug])
    return out
