from __future__ import annotations

import pytest

from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_thought
from orion.schemas.harness_finalize import HarnessTurnOutcomeMoleculeV1, HarnessVerdictMoleculeV1


@pytest.mark.asyncio
async def test_outcome_molecule_before_hub_publish() -> None:
    thought = make_thought()
    appraisal = make_appraisal()
    reflection = make_reflection()
    draft_text = "internal draft"
    final_text = "final reply for juniper"
    events: list[tuple[str, object]] = []

    async def publish_verdict(molecule: HarnessVerdictMoleculeV1, **_: object) -> None:
        events.append(("verdict", molecule))

    async def publish_outcome(molecule: HarnessTurnOutcomeMoleculeV1, **_: object) -> None:
        events.append(("outcome", molecule))

    async def hub_publish_reply() -> None:
        events.append(("hub_reply", None))

    verdict = await emit_verdict_molecule(
        correlation_id="c-1",
        reflection=reflection,
        publish_fn=publish_verdict,
    )
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-1",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        draft_text=draft_text,
        final_text=final_text,
        finalize_changed=True,
        publish_fn=publish_outcome,
    )
    await hub_publish_reply()

    assert [name for name, _ in events] == ["verdict", "outcome", "hub_reply"]
    assert isinstance(outcome, HarnessTurnOutcomeMoleculeV1)
    assert outcome.final_text == final_text
    assert outcome.finalize_changed is True
    assert outcome.verdict_molecule_id
