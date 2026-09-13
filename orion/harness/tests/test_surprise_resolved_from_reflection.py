from __future__ import annotations

import pytest

from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_thought


@pytest.mark.asyncio
async def test_aligned_passthrough_high_surprise_is_resolved() -> None:
    thought = make_thought()
    appraisal = make_appraisal(surprise_level=0.9)
    reflection = make_reflection(alignment_verdict="aligned", strain_unresolved=False)
    verdict = await emit_verdict_molecule(correlation_id="c-1", reflection=reflection)
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-1",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        draft_text="hey. i'm here. what's on your mind?",
        final_text="hey. i'm here. what's on your mind?",
        finalize_changed=False,
    )
    assert outcome.surprise_resolved is True


@pytest.mark.asyncio
async def test_misaligned_is_not_surprise_resolved() -> None:
    thought = make_thought()
    appraisal = make_appraisal(surprise_level=0.01)
    reflection = make_reflection(alignment_verdict="misaligned")
    verdict = await emit_verdict_molecule(correlation_id="c-1", reflection=reflection)
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-1",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        draft_text="draft",
        final_text="draft",
        finalize_changed=False,
        finalize_failed=False,
    )
    assert outcome.surprise_resolved is False
