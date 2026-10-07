"""daily_metacog_v1 prompt fits by construction (redesign Stage 0B).

Regression: from 2026-09-03 the nightly prompt was ~8,467 chars against an
8,192 limit (6,126 of it the JSON skills catalog) and the report failed nightly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = ROOT / "services" / "orion-cortex-exec"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(EXEC_ROOT) not in sys.path:
    sys.path.append(str(EXEC_ROOT))

from app.executor import (  # noqa: E402
    _append_memory_digest,
    _enforce_daily_metacog_prompt_budget,
    _render_prompt,
)
from app.actions_skill_registry import ActionsSkillRegistry  # noqa: E402
from app.settings import Settings as ExecSettings  # noqa: E402
from orion.cognition.daily_metacog_budget import (  # noqa: E402
    DAILY_METACOG_PROMPT_MAX_CHARS,
    _render_simple,
    build_daily_metacog_skill_catalog,
    daily_metacog_digest_max_chars,
)
from orion.cognition.skills_manifest import (  # noqa: E402
    build_compact_skill_catalog,
    load_skill_manifest,
)

TEMPLATE = (ROOT / "orion" / "cognition" / "prompts" / "daily_metacog_prompt.j2").read_text(encoding="utf-8")


def _ctx(*, digest: str, catalog: str, count: int) -> dict:
    return {
        "request_date": "2026-09-30",
        "timezone": "America/Denver",
        "node": "athena",
        "window_start_utc": "2026-09-30T06:00:00+00:00",
        "window_end_utc": "2026-10-01T06:00:00+00:00",
        "memory_digest": digest,
        "skills_catalog_count": count,
        "skills_catalog_compact": catalog,
    }


def _final_prompt(ctx: dict) -> str:
    """What cortex-exec actually measures: render, then _append_memory_digest."""
    prompt = _render_prompt(TEMPLATE, ctx)
    return _append_memory_digest(prompt, ctx["memory_digest"].strip())


def _max_digest() -> str:
    # Realistic multi-line digest padded to the recall profile's hard ceiling.
    n = daily_metacog_digest_max_chars()
    line = "- [sql_timeline:collapse_mirror] metacog draft grounded in recall evidence\n"
    return (line * (n // len(line) + 1))[:n]


def _scaled_manifest(factor: int):
    base = load_skill_manifest()
    out = []
    for i in range(factor):
        for item in base:
            sid = item.skill_id if i == 0 else item.skill_id.replace("skills.", f"skills.copy{i}.", 1)
            out.append(item.model_copy(update={"skill_id": sid}))
    return out


def test_settings_default_matches_shared_budget_constant() -> None:
    # The FIELD default, not the env-loaded value: a local .env override must
    # not make this parity check pass or fail.
    default = ExecSettings.model_fields["daily_metacog_prompt_max_chars"].default
    assert int(default) == DAILY_METACOG_PROMPT_MAX_CHARS


def test_digest_ceiling_is_read_from_recall_profile() -> None:
    assert daily_metacog_digest_max_chars() == 1280


def test_simple_render_matches_cortex_exec_jinja_render() -> None:
    ctx = _ctx(digest="d\nd", catalog="skills.a.v1: x", count=1)
    assert _render_simple(TEMPLATE, ctx) == _render_prompt(TEMPLATE, ctx)


def test_old_json_catalog_with_max_digest_was_over_limit() -> None:
    """Pins the bug: the previous catalog cannot fit beside a full digest."""
    manifest = load_skill_manifest()
    prompt = _final_prompt(_ctx(digest=_max_digest(), catalog=build_compact_skill_catalog(), count=len(manifest)))
    assert len(prompt) > DAILY_METACOG_PROMPT_MAX_CHARS


def test_current_manifest_with_max_digest_fits_and_passes_guard() -> None:
    catalog, count = build_daily_metacog_skill_catalog(load_skill_manifest())
    ctx = _ctx(digest=_max_digest(), catalog=catalog, count=count)
    prompt = _final_prompt(ctx)
    assert len(prompt) <= DAILY_METACOG_PROMPT_MAX_CHARS
    _enforce_daily_metacog_prompt_budget(
        prompt=prompt,
        ctx=ctx,
        correlation_id="test",
        verb_name="daily_metacog_v1",
        step_name="draft_daily_metacog",
    )


def test_every_read_only_skill_id_is_selectable_today() -> None:
    manifest = load_skill_manifest()
    catalog, count = build_daily_metacog_skill_catalog(manifest)
    read_only_ids = {m.skill_id for m in manifest if m.read_only}
    listed = {line.split(":", 1)[0] for line in catalog.splitlines()}
    assert listed == read_only_ids
    assert count == len(read_only_ids)
    # Non-read-only skills are rejected by the daily selector, so never offered.
    assert not ({m.skill_id for m in manifest if not m.read_only} & listed)
    assert all(len(line.split(": ", 1)[1]) <= 60 for line in catalog.splitlines())


@pytest.mark.parametrize("factor", [2, 10])
def test_grown_manifest_still_fits(factor: int) -> None:
    manifest = _scaled_manifest(factor)
    catalog, count = build_daily_metacog_skill_catalog(manifest)
    prompt = _final_prompt(_ctx(digest=_max_digest(), catalog=catalog, count=count))
    assert len(prompt) <= DAILY_METACOG_PROMPT_MAX_CHARS
    read_only_ids = {m.skill_id for m in manifest if m.read_only}
    listed = [line for line in catalog.splitlines() if not line.startswith("(+")]
    listed_ids = {line.split(":", 1)[0] for line in listed}
    assert listed_ids <= read_only_ids
    assert count == len(listed_ids) > 0
    if count < len(read_only_ids):
        assert catalog.splitlines()[-1] == f"(+{len(read_only_ids) - count} more not listed)"


# --- read-only labelling (review finding 4) ---------------------------------

WORLD_CHANGING = ("skills.docker.compose_service_bringup.v1", "skills.imagination.render_scene.v1")


def test_world_changing_skills_are_not_labelled_read_only() -> None:
    by_id = {m.skill_id: m for m in load_skill_manifest()}
    compose = by_id["skills.docker.compose_service_bringup.v1"]
    assert (compose.read_only, compose.idempotent, compose.risk_class) == (False, False, "state_change")
    render = by_id["skills.imagination.render_scene.v1"]
    assert (render.read_only, render.idempotent, render.risk_class) == (False, False, "benign_actuation")


def test_world_changing_skills_are_not_offered_to_metacog() -> None:
    catalog, _count = build_daily_metacog_skill_catalog(load_skill_manifest())
    for sid in WORLD_CHANGING:
        assert sid not in catalog


def test_shared_manifest_and_cortex_exec_registry_classify_every_skill_identically() -> None:
    """The two classifier copies must not drift (builder_prune was fixed in one only)."""
    verbs_dir = ROOT / "orion" / "cognition" / "verbs"
    fields = ("family", "read_only", "idempotent", "risk_class", "requires_confirmation", "requires_execute_opt_in")
    shared = {m.skill_id: tuple(getattr(m, f) for f in fields) for m in load_skill_manifest(verbs_dir=verbs_dir)}
    exec_side = {m.skill_id: tuple(getattr(m, f) for f in fields) for m in ActionsSkillRegistry(verbs_dir=verbs_dir).list()}
    assert shared == exec_side
