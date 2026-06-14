from __future__ import annotations

import sys
import types
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
RECALL_ROOT = ROOT / "services" / "orion-recall"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(RECALL_ROOT) not in sys.path:
    sys.path.append(str(RECALL_ROOT))
if str(RECALL_ROOT / "app") not in sys.path:
    sys.path.append(str(RECALL_ROOT / "app"))

sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logging.getLogger("test")))

from fusion import fuse_candidates  # noqa: E402
from orion.cognition.recall_query import (  # noqa: E402
    recall_ctx_merge_from_reply,
    recall_profile_prompt_flags,
)
from orion.core.contracts.recall import MemoryBundleV1, MemoryItemV1, RecallReplyV1  # noqa: E402


def _load_verb(name: str) -> dict:
    path = ROOT / "orion" / "cognition" / "verbs" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_profile(name: str) -> dict:
    path = ROOT / "orion" / "recall" / "profiles" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _long_candidates(count: int = 12) -> list[dict]:
    chunk = (
        "Scheduler journal lane evidence with long SQL timeline prose about continuity, "
        "hardware diagnostics, ticket threads, and reflective grounding markers "
    )
    return [
        {
            "id": f"sql-{i}",
            "source": "sql_timeline",
            "source_ref": "journal_entry_index",
            "score": 0.9 - (i * 0.01),
            "text": f"item-{i} " + (chunk * 30),
        }
        for i in range(count)
    ]


def test_daily_metacog_verb_uses_metacog_profile() -> None:
    verb = _load_verb("daily_metacog_v1")
    assert verb["recall_profile"] == "journal.daily.metacog.grounded.v1"
    recall_step = next(step for step in verb["steps"] if step["name"] == "recall_context")
    assert recall_step["recall_profile"] == "journal.daily.metacog.grounded.v1"


def test_daily_pulse_verb_uses_scheduler_journal_profile() -> None:
    verb = _load_verb("daily_pulse_v1")
    assert verb["recall_profile"] == "journal.daily.grounded.v1"
    recall_step = next(step for step in verb["steps"] if step["name"] == "recall_context")
    assert recall_step["recall_profile"] == "journal.daily.grounded.v1"


def test_journal_daily_grounded_profile_is_not_strict_by_default() -> None:
    prof = _load_profile("journal.daily.grounded.v1")
    assert prof.get("strict_prompt_budget") is not True
    assert not int(prof.get("render_char_budget") or 0)
    assert not int(prof.get("max_render_snippet_chars") or 0)
    assert prof.get("prompt_safe_ctx") is not True
    flags = recall_profile_prompt_flags("journal.daily.grounded.v1")
    assert flags.get("strict_prompt_budget") is False
    assert flags.get("prompt_safe_ctx") is False


def test_metacog_profile_has_explicit_strict_opt_in_fields() -> None:
    prof = _load_profile("journal.daily.metacog.grounded.v1")
    assert prof.get("strict_prompt_budget") is True
    assert int(prof.get("render_char_budget") or 0) == 1280
    assert int(prof.get("max_render_snippet_chars") or 0) == 280
    assert prof.get("prompt_safe_ctx") is True


def test_scheduler_journal_digest_richer_than_metacog_on_same_items() -> None:
    candidates = _long_candidates()
    journal_profile = _load_profile("journal.daily.grounded.v1")
    metacog_profile = _load_profile("journal.daily.metacog.grounded.v1")
    journal_bundle, _ = fuse_candidates(
        candidates=candidates,
        profile=journal_profile,
        query_text="daily journal continuity",
    )
    metacog_bundle, _ = fuse_candidates(
        candidates=candidates,
        profile=metacog_profile,
        query_text="daily metacog reflection",
    )
    assert journal_bundle.rendered
    assert metacog_bundle.rendered
    assert len(journal_bundle.rendered) > len(metacog_bundle.rendered)
    assert len(journal_bundle.rendered) >= int(len(metacog_bundle.rendered) * 1.25)


def test_journal_daily_grounded_not_capped_at_1280_without_explicit_config() -> None:
    journal_profile = _load_profile("journal.daily.grounded.v1")
    strict_variant = dict(journal_profile)
    strict_variant["strict_prompt_budget"] = True
    strict_variant["render_char_budget"] = 1280
    strict_variant["max_render_snippet_chars"] = 280
    candidates = _long_candidates(count=12)
    default_bundle, _ = fuse_candidates(
        candidates=candidates,
        profile=journal_profile,
        query_text="scheduler daily journal",
    )
    strict_bundle, _ = fuse_candidates(
        candidates=candidates,
        profile=strict_variant,
        query_text="scheduler daily journal",
    )
    assert default_bundle.rendered
    assert strict_bundle.rendered
    assert len(default_bundle.rendered) > len(strict_bundle.rendered)


def test_recall_merge_prompt_safe_ctx_is_opt_in() -> None:
    reply = RecallReplyV1(
        bundle=MemoryBundleV1(
            items=[MemoryItemV1(id="1", source="sql_timeline", snippet="raw snippet body")],
            rendered="digest text",
        )
    )
    default_merge = recall_ctx_merge_from_reply(reply)
    assert "items" in default_merge["memory_bundle"]
    assert "recall_memory_bundle_debug" not in default_merge
    assert "recall_prompt_safe_ctx" not in default_merge

    safe_merge = recall_ctx_merge_from_reply(reply, prompt_safe_ctx=True)
    assert safe_merge["memory_bundle"] == {"rendered": "digest text"}
    assert safe_merge["recall_memory_bundle_debug"]["items"]
    assert safe_merge["recall_prompt_safe_ctx"] is True
