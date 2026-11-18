"""Eval: one full six-hop dream.carry, driven the way orion-durable-runs drives it, through
orion-dream's real prompt builders, reply parser, text handler and finish.

Material: the first real sleep under novelty pressure (dc-981127ddbddf, 2026-10-09) via the
real story_trigger. Pictures: three real captions orion-thought's vision step wrote for
Orion's own paintings (fixtures/painting_captions_2026-10.json). The LLM is a fake that
returns plausible JSON (with an over-long image prompt every time, to prove the clip).

Checks the hand-offs that make the carry a carry rather than three unrelated passages:
each later text prompt contains the previous picture's caption and the previous passage,
the opening prompt contains every replayed item and no hypotheses, every image prompt fits
the CLIP budget, and the finished dream alternates text/image in order, ending on what was seen.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from orion.schemas.dream_carry import (
    IMAGE_PROMPT_MAX_WORDS,
    DreamCarryBriefV1,
    DreamCarryHopV1,
    DreamCarryStepRequestV1,
    dream_carry_run_id,
)

FIXTURES = Path(__file__).parent / "fixtures"
SLEEP = FIXTURES / "sleep_cycle_dc-981127ddbddf_2026-10-09.json"
CAPTIONS = FIXTURES / "painting_captions_2026-10.json"
LEASE = {"lease_id": "eval-lease", "generation": 1, "role": "metacog_background", "holder": "durable:eval"}
# The opening prompt must stay small enough to leave the passage room on a background lane.
MAX_PROMPT_CHARS = 7000


def _sleep_trigger():
    from app.story import story_trigger
    from orion.schemas.dream_cycle import DreamCycleV1

    d = json.loads(SLEEP.read_text())
    cycle = DreamCycleV1(cycle_id=d["cycle_id"], trigger="pressure", status=d["status"], started_at=d["started_at"],
                         ended_at=d["started_at"], pressure=d["pressure"], replay=d["replay"], note=d["note"])
    return cycle, story_trigger(cycle)


def test_six_hop_carry_threads_each_picture_into_the_next_passage():
    from app.carry import FinishLedger, handle_finish, handle_text

    cycle, trigger = _sleep_trigger()
    paintings = json.loads(CAPTIONS.read_text())["paintings"]
    brief = DreamCarryBriefV1(trigger_id=trigger.trigger_id, sleep=trigger.sleep)
    run_id = dream_carry_run_id(trigger.trigger_id)
    prompts: list[str] = []

    async def fake_llm(prompt, gpu_lease, timeout_sec):
        assert gpu_lease == LEASE
        prompts.append(prompt)
        n = len(prompts)
        scene = " ".join(["a lamp-lit porch"] + [f"detail{i}" for i in range(IMAGE_PROMPT_MAX_WORDS + 10)])
        return "```json\n" + json.dumps({"passage": f"Passage {n}. The dream moves on.", "image_prompt": scene}) + "\n```"

    hops: list[DreamCarryHopV1] = []
    for index in range(brief.hops):
        if index % 2 == 0:
            req = DreamCarryStepRequestV1(run_id=run_id, correlation_id="c-eval", step="text", brief=brief,
                                          hops=hops, hop_index=index, gpu_lease=LEASE)
            result = asyncio.run(handle_text(req, fake_llm))
            assert result.status == "done", result.reason
            hops.append(result.hop)
        else:  # the image hop orion-thought + durable-runs make: paint, then see
            p = paintings[index // 2]
            hops.append(DreamCarryHopV1(kind="image", index=index, sha256=p["sha256"], caption=p["caption"],
                                        child_run_id=p["durable_run_id"]))

    assert len(prompts) == 3
    opening = prompts[0]
    print(f"\nopening prompt {len(opening)} chars; later prompts {[len(p) for p in prompts[1:]]}")
    for r in cycle.replay:
        assert " ".join(r.text.split())[:60] in opening
    assert "hypothes" not in opening.lower() and len(opening) <= MAX_PROMPT_CHARS
    for k in (1, 2):
        later = prompts[k]
        assert f"looking at it you see: {paintings[k - 1]['caption']}" in later
        assert f"Passage {k}. The dream moves on." in later
        assert "TONIGHT'S SLEEP" not in later
    for h in hops:
        if h.kind == "text":
            assert len(h.image_prompt.split()) == IMAGE_PROMPT_MAX_WORDS

    published = []

    async def publish(dream):
        published.append(dream)

    done = asyncio.run(handle_finish(
        DreamCarryStepRequestV1(run_id=run_id, correlation_id="c-eval", step="finish", brief=brief, hops=hops),
        publish, FinishLedger()))
    assert done.status == "done"
    (dream,) = published
    assert dream.dream_id == done.dream_id
    assert [f["kind"] for f in dream.fragments] == ["text", "image"] * 3
    assert [f["index"] for f in dream.fragments] == list(range(6))
    assert [f["sha256"] for f in dream.fragments if f["kind"] == "image"] == [p["sha256"] for p in paintings]
    assert dream.narrative.endswith(f"[picture] {paintings[2]['caption']}")  # what was seen is the last word
    assert dream.trigger["sleep"]["cycle_id"] == cycle.cycle_id and dream.trigger["carry_run_id"] == run_id
