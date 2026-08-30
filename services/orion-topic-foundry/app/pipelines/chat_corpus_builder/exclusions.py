"""Operator-curated chat turns kept out of topic-foundry's training corpus.

See config/corpus/topic_foundry_excluded_turns.yaml for the full rationale.
The short version: AI Town material reaches the main corpus through Orion's
own turns, the rows carry no marker distinguishing them from ordinary
conversation, and the corpus is small enough (273 rows on 2026-08-30) that two
stray turns produced a top-three concept by betweenness.

This module only decides what training READS. It never deletes anything: the
rows stay in Postgres, and removing an id from the config restores them on the
next run.

FAILS OPEN, DELIBERATELY. A missing or malformed config yields an empty
exclusion set and training proceeds on the full corpus. The alternative --
refusing to train -- turns a config typo into a silent halt of the whole
induction pipeline, which is a worse failure than re-learning a concept an
operator wanted dropped. Every degraded path logs, so "no exclusions applied"
is never silent.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo-root-relative by default; overridable so a test or a differently-mounted
# container can point at its own file without patching module internals.
_DEFAULT_RELATIVE_PATH = "config/corpus/topic_foundry_excluded_turns.yaml"
_ENV_VAR = "TOPIC_FOUNDRY_EXCLUDED_TURNS_PATH"


def _config_path() -> Path:
    """Explicit env var first, then an upward search for the config.

    NOT a fixed `parents[N]`, because the repo and the container do not agree
    on depth. In the repo this file is
    `services/orion-topic-foundry/app/pipelines/chat_corpus_builder/exclusions.py`
    (repo root is 5 up); in the image it is `/app/app/pipelines/...` (only 3
    up, and `parents[5]` clamps to `/`). A hardcoded index therefore resolves
    to `/config/corpus/...` inside the container, finds nothing, and fails
    open -- the exclusions would silently never apply in the one place they
    matter. Walking up for the file works in both layouts.
    """
    override = str(os.getenv(_ENV_VAR, "")).strip()
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _DEFAULT_RELATIVE_PATH
        if candidate.exists():
            return candidate
    # Nothing found: return the repo-shaped path so the log names something real.
    return here.parents[min(5, len(here.parents) - 1)] / _DEFAULT_RELATIVE_PATH


def load_excluded_turn_ids(path: Path | None = None) -> frozenset[str]:
    """The curated id set, or an empty set if it cannot be read."""
    target = path or _config_path()
    try:
        import yaml

        raw = yaml.safe_load(target.read_text()) or {}
    except FileNotFoundError:
        logger.info("topic_foundry_exclusions_absent path=%s", target)
        return frozenset()
    except Exception as exc:  # noqa: BLE001 - a config typo must not halt training
        logger.warning("topic_foundry_exclusions_unreadable path=%s error=%s", target, exc)
        return frozenset()

    entries = raw.get("excluded_turn_ids") or []
    if not isinstance(entries, list):
        logger.warning("topic_foundry_exclusions_malformed path=%s type=%s", target, type(entries).__name__)
        return frozenset()

    ids: set[str] = set()
    for entry in entries:
        # Accept both {id: ...} and a bare string, so a hand-edit that drops
        # the reason field still applies rather than silently doing nothing.
        value = entry.get("id") if isinstance(entry, dict) else entry
        text = str(value or "").strip()
        if text:
            ids.add(text)
    if ids:
        logger.info("topic_foundry_exclusions_loaded count=%d path=%s", len(ids), target)
    return frozenset(ids)


@lru_cache(maxsize=1)
def cached_excluded_turn_ids() -> frozenset[str]:
    """Process-lifetime cache. Training runs are long and re-reading the file
    per query would let the corpus change under a single run."""
    return load_excluded_turn_ids()
