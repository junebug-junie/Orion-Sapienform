"""Offline eval: the rendered stance prompt for a reading turn holds one output contract.

Renders the real ``stance_react.j2`` the way a reading turn feeds it (the stance
view as ``user_message`` and ``stance_inputs.user_message``) for Stage 1, a
document Stage 1, and a Stage 2 with a retry note and claim context. The only
output keys the stance may see are its own (ThoughtEventV1); the one exception is the
labelled Stage 1 handoff, which is input the stance must read. Live 2026-10-27/28
the reader's keys in this prompt were what Qwen3.6-35B wrote instead (6 of 6
reading stances on that model) and what Bonsai-2-27B spent its whole token
budget weighing (empty stance replies).

This checks the prompt, not live model quality. Live quality: run
``python -m orion.thought.evals.stance_task_boundary`` with a stance view as the
task file (opt-in, uses the GPU pool).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment

from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1
from orion.world_pulse_read.reader_contract import stance_view_of_reading_prompt
from scripts.world_pulse_read_pipeline import _build_document_stage1_prompt, _build_stage1_prompt
from scripts.world_pulse_read_stage2 import _build_stage2_prompt

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = (REPO / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")

# Reader output keys that must never reach the stance pass. Quoted, so a word in the
# task prose ("note hops") does not count.
READER_KEYS = (
    '"what_i_learned"', '"candidate_priors"', '"concept_candidates"', '"open_threads"',
    '"summary"', '"article_summary"', '"priors_tested"', '"hops"', '"need_stage1_urls"',
    '"relationship_claims"',
)

SEED = WorldPulseReadSeedV1(
    seed_id="reading:eval", kind="reading", run_id="eval", url="https://example.org/paper",
)
DOC_SEED = WorldPulseReadSeedV1(
    seed_id="reading:doc", kind="reading", run_id="eval", url="file:///docs/spec.md",
)


def _stage2_prompt() -> str:
    from datetime import datetime, timezone

    from orion.world_pulse_read.assertions import ClaimContextV1, EndpointV1
    from orion.world_pulse_read.fetch_text import RetainedTextV1

    handoff = WorldPulseReadHandoffV1(
        seed_ref=SEED, what_i_learned="Stage 1 read the abstract only.", trace_id="tr-1",
        created_at=datetime.now(timezone.utc),
    )
    claims = ClaimContextV1(
        subjects=(EndpointV1(node_id="sub-1", kind="concept", label="heat pump"),),
        objects=(EndpointV1(node_id="obj-1", kind="concept", label="refrigeration cycle"),),
        texts=(RetainedTextV1(sha256="c" * 64, url=SEED.url, representation="tool_digest",
                              text="A heat pump uses a refrigeration cycle."),),
    )
    return _build_stage2_prompt(
        handoff, "tr-2", claim_context=claims, prior_error="turn_exception:TimeoutError: ",
    )


PROMPTS = {
    "stage1": _build_stage1_prompt(SEED, "tr-1"),
    "document_stage1": _build_document_stage1_prompt(DOC_SEED, "tr-1", sha256="b" * 64, text="Spec body."),
    "stage2": _stage2_prompt(),
}


def _render(task: str) -> str:
    return Environment().from_string(TEMPLATE).render(
        user_message=task, stance_inputs={"user_message": task}, association={},
        repair_bundle=None, coalition_projection=None, self_prediction_enabled=True,
    )


# Stage 2 shows the stance the Stage 1 handoff (a dict repr, single-quoted keys). It is
# input the stance must read, so it stays, on one line, labelled as input data. Only
# that line is exempt from the single-quoted check.
HANDOFF_LABEL = "Stage 1 handoff (input data from the earlier read, not an output shape): handoff="


def _reader_keys_in(rendered: str) -> list[str]:
    # The template renders user_message with |tojson: a key arrives as \"summary\",
    # a newline as \n, a single quote as \u0027. A dict repr would carry 'summary'.
    lines = rendered.replace("\\n", "\n").replace("\\u0027", "'").split("\n")
    outside_handoff = "\n".join(line for line in lines if HANDOFF_LABEL not in line)
    found = []
    for k in READER_KEYS:
        bare = k.strip('"')
        if k in rendered or k.replace('"', '\\"') in rendered or f"'{bare}'" in outside_handoff:
            found.append(k)
    return found


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_reading_stance_prompt_holds_only_the_stance_contract(name):
    full = PROMPTS[name]
    view = stance_view_of_reading_prompt(full)
    assert view is not None, f"{name}: reader contract is not fenced"
    before = _reader_keys_in(_render(full))
    after = _reader_keys_in(_render(view))
    print(f"reading_stance_prompt_eval {name} reader_keys_before={len(before)} after={len(after)}")
    assert before, f"{name}: the full prompt should carry reader keys (fixture sanity)"
    assert after == [], f"{name}: reader keys leak into the stance prompt: {after}"
    rendered = _render(view)
    assert "REQUIRED JSON FIELDS" in rendered and "- imperative:" in rendered
    # The reader's output-format orders are a second contract too.
    for order in ("```json", "JSON object", "Juniper-facing prose"):
        assert order not in view, f"{name}: {order!r} outside the fence"
    if name == "stage2":
        assert view.count(HANDOFF_LABEL) == 1
        assert "relationship_claims" in full and "relationship_claims" not in view
        assert "A heat pump uses a refrigeration cycle." in view  # retained source text stays
