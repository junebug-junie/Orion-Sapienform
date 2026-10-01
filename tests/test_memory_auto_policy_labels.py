"""Stage 0A labels: nothing tells Orion that Juniper approved an auto-saved memory.

Live 2026-09-30: 706 of 743 active crystallizations were auto-saved by the
formation policy (`governance.approval_mode='auto_policy'`) and nobody
reviewed them. Three surfaces described them as approved/accepted: the
self-study source description, the curiosity menu cards (no per-row label at
all), and the curiosity journal footer ("N approved concepts").
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from orion.curiosity.journal import MaterialCounts, build_investigation_journal_entry
from orion.curiosity.study_material import APPROVED_SAMPLE_SQL, assemble_study_material

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[1]


def _row(cid, mode, approved=False):
    return {
        "juniper_approved": approved,
        "crystallization_id": cid,
        "kind": "semantic",
        "subject": "Run github compactor.",
        "summary": "Run github compactor.",
        "salience": 0.4,
        "created_at": NOW,
        "approval_mode": mode,
    }


def _material(rows):
    return assemble_study_material(
        now=NOW,
        approved_counts=[{"kind": "semantic", "n": 2, "manual_n": 1}],
        approved_rows=rows,
        relation_counts=[],
        relation_rows=[],
    )


def test_auto_policy_card_says_auto_saved_not_approved():
    card = _material([_row("a", "auto_policy")]).crystallizations[0]
    first_line = card.preview().splitlines()[0]
    assert "auto-saved by policy" in first_line
    assert "not reviewed by Juniper" in first_line
    assert "approved by Juniper" not in first_line


def test_hand_approved_card_says_approved():
    card = _material([_row("m", "manual_required", approved=True)]).crystallizations[0]
    assert "approved by Juniper" in card.preview()


def test_manual_required_without_an_approve_row_is_not_called_approved():
    # Review of PR #2457: approval_mode says a row NEEDED review, not that it
    # got one. Only an op='approve' history row counts.
    card = _material([_row("m", "manual_required", approved=False)]).crystallizations[0]
    assert "approved by Juniper" not in card.preview()
    assert "no recorded approval" in card.preview()


def test_approval_is_read_from_history_in_both_queries():
    from orion.curiosity.study_material import APPROVED_COUNT_SQL

    for sql in (APPROVED_COUNT_SQL, APPROVED_SAMPLE_SQL):
        assert "memory_crystallization_history" in sql and "h.op = 'approve'" in sql
    assert "<> 'auto_policy'" not in APPROVED_COUNT_SQL


def test_card_without_the_column_claims_nothing():
    row = _row("x", None)
    del row["approval_mode"]
    del row["juniper_approved"]
    card = _material([row]).crystallizations[0]
    assert card.approval_label is None
    assert "approved" not in card.preview()
    assert "auto-saved" not in card.preview()


def test_sample_sql_selects_the_approval_mode():
    assert "approval_mode" in APPROVED_SAMPLE_SQL


def test_journal_footer_says_saved_not_approved():
    entry = build_investigation_journal_entry(
        material=MaterialCounts(
            approved_total=40,
            approved_by_kind={"semantic": 40},
            crystallization_count=12,
            relation_total=0,
            relation_count=0,
        ),
        body_text="Orion looked around.",
        correlation_id="00000000-0000-0000-0000-000000000001",
        run_id="run-1",
    )
    assert "40 saved concepts" in entry.body
    assert "approved concepts" not in entry.body


def _self_study_module():
    path = REPO / "services" / "orion-cortex-exec" / "app" / "self_study_analysis.py"
    spec = importlib.util.spec_from_file_location("_self_study_analysis_label_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_self_study_source_description_does_not_claim_approval():
    text = _self_study_module().SOURCE_SPECS["concept_induction"].what_it_measures
    assert "accepted" not in text
    assert "auto-saved by policy" in text
    assert "does not mean Juniper approved" in text
    assert "'approve'" in text  # approval is a history row, not an inference
