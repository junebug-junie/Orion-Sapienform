"""A wrong-shaped draft field must not discard a usable summary.

Live 2026-09-29: baseline drafts put biometrics into what_changed.evidence (a
dict). MetacogDraftTextPatchV1 still declared what_changed, so the whole draft --
summary and mantra included -- failed validation, and the baseline firebreak
dropped ~95% of baseline rows. what_changed is publish-computed from evidence,
so the draft schema no longer declares it (the sanitizer strips it), and
validation drops only the fields that fail.
"""
from __future__ import annotations

from app.executor import _sanitize_patch_payload, _validate_draft_patch_per_field
from orion.schemas.metacog_patches import MetacogDraftTextPatchV1


def test_what_changed_is_stripped_not_validated():
    raw = {
        "summary": "Nothing moved this hour; the check found no event.",
        "mantra": "Quiet is data too.",
        "what_changed": {"evidence": {"biometrics": {"status": "fresh"}}},
    }
    sanitized, stripped = _sanitize_patch_payload(raw, model=MetacogDraftTextPatchV1)
    assert "what_changed" in stripped
    patch, invalid = _validate_draft_patch_per_field(sanitized)
    assert invalid == []
    assert patch is not None and patch.summary.startswith("Nothing moved")


def test_one_wrong_shaped_field_keeps_the_rest():
    patch, invalid = _validate_draft_patch_per_field(
        {"summary": "A gateway timeout, then recovery.", "mantra": "Hold.", "tags_suggested": "state:steady"}
    )
    assert invalid == ["tags_suggested"]
    assert patch is not None
    assert (patch.summary, patch.mantra, patch.tags_suggested) == ("A gateway timeout, then recovery.", "Hold.", None)


def test_nothing_usable_is_a_real_fallback():
    patch, invalid = _validate_draft_patch_per_field({"summary": {"not": "text"}, "tags_suggested": ["x"]})
    assert patch is None
    assert invalid == ["summary"]


def test_only_a_stripped_what_changed_is_a_fallback_not_an_empty_llm_draft():
    # Review finding: {"what_changed": ...} alone sanitized to {} and validated
    # as an empty "llm" patch, publishing the fallback template text as cognition.
    sanitized, _ = _sanitize_patch_payload(
        {"what_changed": {"evidence": {"a": 1}}}, model=MetacogDraftTextPatchV1
    )
    patch, invalid = _validate_draft_patch_per_field(sanitized)
    assert patch is None
    assert invalid == ["summary:missing"]


def test_mantra_without_summary_is_a_fallback():
    patch, invalid = _validate_draft_patch_per_field({"summary": {"x": 1}, "mantra": "Hold."})
    assert patch is None
    assert invalid == ["summary"]
