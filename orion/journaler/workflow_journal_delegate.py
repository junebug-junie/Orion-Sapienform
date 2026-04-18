"""Shared rules for workflow-owned journal persistence (orch vs exec)."""

from __future__ import annotations

from typing import Any, Mapping


def workflow_journal_write_delegated_to_exec(metadata: Mapping[str, Any] | None) -> bool:
    """
    When True, journal.entry.write.v1 is emitted from cortex-exec after journal.compose
    (LegacyPlanVerb) so the write uses the same connected bus as other exec-side publishers.
    Cortex-orch must skip its duplicate publish for the same request.
    """
    if not isinstance(metadata, dict):
        return False
    wf = metadata.get("workflow_execution") or {}
    wf_id = str(metadata.get("workflow_id") or (wf.get("workflow_id") if isinstance(wf, dict) else "") or "").strip()
    sub = str(
        metadata.get("workflow_subverb") or (wf.get("workflow_subverb") if isinstance(wf, dict) else "") or ""
    ).strip()
    return wf_id in ("journal_pass", "journal_discussion_window_pass") and sub == "journal.compose"
