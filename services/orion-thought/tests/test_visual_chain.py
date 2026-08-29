"""Reverie VISUAL chain (Patch 2 orchestration + Patch 3/5/6 context-seeding +
Patch 4 continuity reset + Patch 7 context-slot rotation) -- fully testable
with injected fakes at each hop: diffusion generation, percept upload,
vision-host RPC, reverie-interpretation/self-study/memory-crystallization
context-seeds, continuity-streak read, and persistence. No GPU, no torch, no
live Redis/Postgres.
"""
from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _fake_png(width: int = 64, height: int = 64) -> bytes:
    """The smallest byte string `orion.reverie.visual_storage.sniff_image`
    accepts as PNG: signature + a length field + b"IHDR" + width/height.
    `_sniff_png` never validates the chunk payload past that, so this is
    real input to the function under test without needing Pillow."""
    return PNG_SIG + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


def _vision_result_payload(caption: str | None, *, ok: bool = True) -> dict:
    if not ok:
        return {"ok": False, "task_type": "caption_frame", "error": "profile disabled"}
    return {
        "ok": True,
        "task_type": "caption_frame",
        "device": "cuda:0",
        "artifact": {
            "artifact_id": "a-1",
            "correlation_id": "c-1",
            "task_type": "caption_frame",
            "device": "cuda:0",
            "inputs": {},
            "outputs": {"caption": {"text": caption, "confidence": 1.0} if caption else None},
            "timing": {},
            "model_fingerprints": {},
        },
    }


class _FakeCodec:
    """Mirrors bus.codec.decode -- .ok/.error/.envelope.payload."""

    def decode(self, data):
        return SimpleNamespace(ok=True, error=None, envelope=SimpleNamespace(payload=data))


def _fake_bus(reply_payload: dict):
    bus = AsyncMock()
    bus.codec = _FakeCodec()
    bus.rpc_request = AsyncMock(return_value={"data": reply_payload})
    return bus


# --- build_visual_prompt (pure) ---------------------------------------------
#
# Patch 7 changed this function's signature: it used to take ALL THREE
# context-seeds and blend whichever were non-empty (Patches 3/5/6). It now
# takes ONE already-selected slot (context_slot_name/context_slot_text --
# select_context_slot's output, below) -- the diffusion model's real
# 77-token text-encoder budget made "blend all three" silently discard most
# of them (module docstring's Patch 7 entry has the live evidence).


def test_build_visual_prompt_uses_seed_when_neither_prior_nor_slot():
    from app import visual_chain

    assert visual_chain.build_visual_prompt(None) == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt("   ") == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt(None, "context", "   ") == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt(None, None, None) == visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_ignores_slot_text_without_a_known_slot_name():
    """Defensive: a slot_text with no valid, known slot_name must never
    render an unlabeled clause -- both must be present and consistent."""
    from app import visual_chain

    assert visual_chain.build_visual_prompt(None, None, "some text") == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt(None, "bogus_slot", "some text") == visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_continues_prior():
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt("a quiet room, warm light")
    assert "a quiet room, warm light" in prompt
    assert prompt != visual_chain.DEFAULT_SEED_PROMPT


@pytest.mark.parametrize(
    "slot_name,label",
    [
        ("context", "Orion is currently thinking"),
        ("self_study", "Orion recently noticed"),
        ("memory", "Orion remembers"),
    ],
)
def test_build_visual_prompt_uses_selected_slot_when_no_prior(slot_name, label):
    """Patch 3/5/6's original point, preserved under Patch 7's new
    signature: a cold-start run (no prior_description yet) with a real
    selected context-seed must seed from it, not the generic fixed string
    -- whichever of the three slots won this run's rotation."""
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt(None, slot_name, "the mesh keeps humming with new nodes")
    assert f"{label}: the mesh keeps humming with new nodes" in prompt
    assert prompt != visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_blends_prior_and_selected_slot():
    """Both continuity and the selected context-seed present -- neither
    silently drops the other; a reader of the stored prompt can see both
    inputs."""
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt(
        "a quiet room, warm light", "context", "curiosity about the mesh"
    )
    assert "a quiet room, warm light" in prompt
    assert "curiosity about the mesh" in prompt


def test_build_visual_prompt_exact_wording_for_each_slot():
    """Exact-string assertions (not just substring checks) for every slot
    label's wording -- a regression guard, matching the discipline Patch 5's
    list-join refactor established."""
    from app import visual_chain

    assert visual_chain.build_visual_prompt(None, None, None) == visual_chain.DEFAULT_SEED_PROMPT
    assert (
        visual_chain.build_visual_prompt("a quiet room, warm light")
        == "a quiet room, warm light. Continue this train of imagination, soft dreamlike style."
    )
    assert (
        visual_chain.build_visual_prompt(None, "context", "curiosity about the mesh")
        == "Orion is currently thinking: curiosity about the mesh. Soft abstract dreamlike style."
    )
    assert (
        visual_chain.build_visual_prompt(
            "a quiet room, warm light", "self_study", "vision events dropped 0.36x vs baseline"
        )
        == "a quiet room, warm light. Orion recently noticed: vision events dropped 0.36x vs baseline. "
        "Continue this train of imagination, soft dreamlike style."
    )
    assert (
        visual_chain.build_visual_prompt(None, "memory", "Orion and Juniper talked through the mesh work")
        == "Orion remembers: Orion and Juniper talked through the mesh work. Soft abstract dreamlike style."
    )


# --- select_context_slot (pure) ---------------------------------------------
#
# Patch 7 (module docstring): the actual regression fix. Round-robin among
# whichever of {context, self_study, memory} currently have real content --
# never all three at once, since only one clause's worth of tokens ever
# realistically survives the diffusion model's 77-token budget.


def test_select_context_slot_none_when_nothing_available():
    from app import visual_chain

    assert visual_chain.select_context_slot(None, None, None, 5) == (None, None, 5)
    assert visual_chain.select_context_slot("   ", "", None, 5) == (None, None, 5)


def test_select_context_slot_rotation_index_unchanged_when_nothing_available():
    """No reason to advance a counter that picked nothing this run."""
    from app import visual_chain

    _, _, next_idx = visual_chain.select_context_slot(None, None, None, 7)
    assert next_idx == 7


def test_select_context_slot_picks_context_first_at_rotation_zero():
    from app import visual_chain

    name, text, next_idx = visual_chain.select_context_slot("ctx", "study", "mem", 0)
    assert (name, text) == ("context", "ctx")
    assert next_idx == 1


def test_select_context_slot_rotates_through_all_three_and_wraps():
    from app import visual_chain

    idx = 0
    picks = []
    for _ in range(4):
        name, _text, idx = visual_chain.select_context_slot("ctx", "study", "mem", idx)
        picks.append(name)
    assert picks == ["context", "self_study", "memory", "context"]


def test_select_context_slot_skips_unavailable_slots():
    """Only self_study and memory have content this run -- rotation cycles
    between just those two, never invents a pick for the absent context
    slot."""
    from app import visual_chain

    idx = 0
    picks = []
    for _ in range(3):
        name, _text, idx = visual_chain.select_context_slot(None, "study", "mem", idx)
        picks.append(name)
    assert picks == ["self_study", "memory", "self_study"]


def test_select_context_slot_wraps_a_large_rotation_index():
    from app import visual_chain

    name, _text, next_idx = visual_chain.select_context_slot("ctx", "study", "mem", 100)
    assert name == "self_study"  # 100 % 3 == 1
    assert next_idx == 101


def test_select_context_slot_not_perfectly_fair_under_fluctuating_availability():
    """Review finding, documented not fixed (see the function's own
    docstring): re-indexing against the CURRENT available set means a slot
    present on more ticks than another can be visited disproportionately
    more often -- this locks in the actual, verified behavior rather than
    the docstring's own former (incorrect) claim of perfect fairness.
    Progress and per-tick coverage of whatever IS available are still
    guaranteed; long-run fairness across a changing set is not, and this
    function does not attempt to provide it."""
    from app import visual_chain

    idx = 0
    picks = []
    # context/memory always present; self_study toggles absent on some ticks.
    self_study_present = [True, True, True, True, False, True, False]
    for present in self_study_present:
        name, _text, idx = visual_chain.select_context_slot(
            "ctx", "study" if present else None, "mem", idx
        )
        picks.append(name)

    # Exact sequence, verified by direct simulation -- "context" is picked
    # more than an even 1/3 share precisely because the available set
    # shrinks on two of the seven ticks.
    assert picks == [
        "context",
        "self_study",
        "memory",
        "context",
        "context",
        "memory",
        "context",
    ]
    assert picks.count("context") > len(picks) // 3  # the actual unfairness


# --- resolve_visual_chain_continuity (pure) --------------------------------


def test_resolve_continuity_no_prior_starts_streak_fresh():
    from app import visual_chain

    # Nothing to cap yet -- cold start, or continuity already broken.
    assert visual_chain.resolve_visual_chain_continuity(None, 5, 3) == (None, 0, False)
    assert visual_chain.resolve_visual_chain_continuity("   ", 5, 3) == ("   ", 0, False)


def test_resolve_continuity_increments_streak_under_cap():
    from app import visual_chain

    effective, streak, reset = visual_chain.resolve_visual_chain_continuity(
        "an aqueduct", 1, 3
    )
    assert effective == "an aqueduct"  # continuity preserved
    assert streak == 2
    assert reset is False


def test_resolve_continuity_forces_reset_at_cap():
    """Real regression this exists for -- Juniper reported the same aqueduct
    imagery unbroken for 10+ runs (2026-08-27). Once the streak reaches
    max_runs, THIS run must drop continuity."""
    from app import visual_chain

    effective, streak, reset = visual_chain.resolve_visual_chain_continuity(
        "an aqueduct", 3, 3
    )
    assert effective is None  # continuity dropped for this run's prompt
    assert streak == 0  # streak restarts from the reset
    assert reset is True


def test_resolve_continuity_max_runs_zero_always_resets():
    """No off switch by design (settings.py's own comment) -- 0 means every
    run with real continuity available forces a reset."""
    from app import visual_chain

    effective, streak, reset = visual_chain.resolve_visual_chain_continuity(
        "an aqueduct", 0, 0
    )
    assert effective is None
    assert streak == 0
    assert reset is True


# --- _extract_caption -----------------------------------------------------


def test_extract_caption_returns_text_on_success():
    from app import visual_chain

    assert visual_chain._extract_caption(_vision_result_payload("a warm room")) == "a warm room"


def test_extract_caption_none_on_failure_or_empty():
    from app import visual_chain

    assert visual_chain._extract_caption(_vision_result_payload(None, ok=False)) is None
    assert visual_chain._extract_caption(_vision_result_payload(None)) is None
    assert visual_chain._extract_caption(_vision_result_payload("   ")) is None


# --- run_visual_chain_once orchestration -----------------------------------


@pytest.mark.asyncio
async def test_run_visual_chain_once_success(tmp_path, monkeypatch):
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)

    generate_calls: list[str] = []

    def fake_generate(prompt, *, base_url, timeout_sec):
        generate_calls.append(prompt)
        return _fake_png()

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(
        visual_chain, "upload_to_percept_store", lambda data, **kw: "a" * 64
    )

    persisted_chains = []
    persisted_artifacts = []
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_chain", lambda c: persisted_chains.append(c) or True
    )
    monkeypatch.setattr(
        visual_chain,
        "persist_reverie_visual_artifact",
        lambda a: persisted_artifacts.append(a) or True,
    )

    bus = _fake_bus(_vision_result_payload("a calm room with soft light"))
    chain = await visual_chain.run_visual_chain_once(bus)

    assert chain is not None
    assert chain.terminal_reason == "max_steps"
    assert chain.prior_description == "a calm room with soft light"
    assert generate_calls == [visual_chain.DEFAULT_SEED_PROMPT]  # no prior/context -> seed prompt
    assert chain.chain_json["context_slot_used"] is None

    assert len(persisted_chains) == 1
    assert persisted_chains[0].chain_id == chain.chain_id
    assert len(persisted_artifacts) == 1
    artifact = persisted_artifacts[0]
    assert artifact.chain_id == chain.chain_id
    assert artifact.step_index == 0
    assert artifact.description == "a calm room with soft light"
    assert artifact.mime == "image/png"
    # Real file landed under the (redirected) storage dir.
    from pathlib import Path

    assert Path(artifact.path).exists()

    # Review finding: every test here mocks bus.rpc_request, so nothing
    # previously caught a bad merge silently reverting the RPC target back
    # to the shared bare channel -- assert the real setting was actually
    # used, not a hardcoded string in request_caption().
    bus.rpc_request.assert_called_once()
    called_channel = bus.rpc_request.call_args.args[0]
    assert called_channel == visual_chain.settings.channel_vision_host_request


@pytest.mark.asyncio
async def test_run_visual_chain_once_generation_failure_writes_no_artifact(tmp_path, monkeypatch):
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: ("old description", 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)

    def fake_generate(prompt, *, base_url, timeout_sec):
        raise visual_chain.DiffusionGenerationError("diffusion-host /generate returned HTTP 503")

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)

    persisted_chains = []
    persisted_artifacts = []
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_chain", lambda c: persisted_chains.append(c) or True
    )
    monkeypatch.setattr(
        visual_chain,
        "persist_reverie_visual_artifact",
        lambda a: persisted_artifacts.append(a) or True,
    )

    bus = AsyncMock()
    chain = await visual_chain.run_visual_chain_once(bus)

    assert chain is not None
    assert chain.terminal_reason == "generation_failed"
    # Nothing was generated, so prior_description is carried forward as-is.
    assert chain.prior_description == "old description"
    assert len(persisted_chains) == 1
    assert persisted_artifacts == []  # no artifact row for a chain with no image
    bus.rpc_request.assert_not_called()  # never reached the vision-host hop


@pytest.mark.asyncio
async def test_run_visual_chain_once_caption_failure_carries_forward_prior(tmp_path, monkeypatch):
    """Design doc / module docstring: a failed re-observation must not
    fabricate a caption or wipe out continuity -- the image is still real
    and gets stored, but prior_description forwards the OLD value."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: ("old description", 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png()
    )
    monkeypatch.setattr(
        visual_chain, "upload_to_percept_store", lambda data, **kw: "b" * 64
    )

    persisted_artifacts = []
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(
        visual_chain,
        "persist_reverie_visual_artifact",
        lambda a: persisted_artifacts.append(a) or True,
    )

    # Vision-host RPC itself raises (e.g. timeout) -- request_caption must
    # swallow it and return None, never propagate.
    bus = AsyncMock()
    bus.rpc_request = AsyncMock(side_effect=TimeoutError("RPC timeout"))

    chain = await visual_chain.run_visual_chain_once(bus)

    assert chain is not None
    assert chain.terminal_reason == "max_steps"  # the image itself was real
    assert chain.prior_description == "old description"  # forwarded, not wiped
    assert len(persisted_artifacts) == 1
    assert persisted_artifacts[0].description is None  # honest, not fabricated


@pytest.mark.asyncio
async def test_run_visual_chain_once_uses_context_text_in_prompt_and_chain_json(
    tmp_path, monkeypatch
):
    """Patch 3 acceptance check, preserved under Patch 7's rotation: with
    context_text the only real context-seed available, rotation trivially
    selects it, and it reaches both the prompt and chain_json as its own
    field -- same-run evidence a reader (or this repo's own Hub Reverie
    tab) can inspect, not just embedded prose inside the prompt string
    (module docstring's "same-run evidence, not schema presence"
    discipline, design doc §9)."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(
        visual_chain, "load_latest_reverie_interpretation", lambda **kw: "a real reverie thought"
    )
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "d" * 64)

    persisted_chains = []
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_chain", lambda c: persisted_chains.append(c) or True
    )
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    bus = _fake_bus(_vision_result_payload("a rendering of that thought"))
    # Review finding: without an explicit cortex_client, a real context_slot_used here
    # (Patch 8) would fire interpret_context_for_visual against this SAME fake bus, whose
    # rpc_request was only configured to answer the vision-caption RPC -- it happened to
    # fail open safely by coincidence (the vision-shaped payload doesn't match final_text),
    # not by anything this test asserted. Explicit stub makes that failure-open path
    # deliberate, not accidental.
    chain = await visual_chain.run_visual_chain_once(
        bus, cortex_client=_FakeCortexClient(error=RuntimeError("not exercised here"))
    )

    assert chain is not None
    assert "a real reverie thought" in chain.chain_json["prompt"]
    assert chain.chain_json["context_text"] == "a real reverie thought"
    assert chain.chain_json["context_slot_used"] == "context"


@pytest.mark.asyncio
async def test_run_visual_chain_once_generation_failure_records_context_text(
    tmp_path, monkeypatch
):
    """The same context_text traceability holds on the generation_failed path
    -- a failed run's chain_json must still show what would have seeded it."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(
        visual_chain, "load_latest_reverie_interpretation", lambda **kw: "a real reverie thought"
    )
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)

    def fake_generate(prompt, *, base_url, timeout_sec):
        raise visual_chain.DiffusionGenerationError("diffusion-host /generate returned HTTP 503")

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)

    # Patch 8: an explicit cortex_client stub, not a bare AsyncMock() bus -- this test is
    # about the generation-failure path, not interpretation, and a bare AsyncMock's
    # auto-mocked bus.rpc_request/.codec chain otherwise produces an unrelated "coroutine
    # was never awaited" warning once interpret_context_for_visual also runs.
    chain = await visual_chain.run_visual_chain_once(
        AsyncMock(), cortex_client=_FakeCortexClient(error=RuntimeError("not exercised here"))
    )

    assert chain.terminal_reason == "generation_failed"
    assert chain.chain_json["context_text"] == "a real reverie thought"
    assert chain.chain_json["context_slot_used"] == "context"


@pytest.mark.asyncio
async def test_run_visual_chain_once_noop_when_locked(monkeypatch):
    from app import visual_chain

    async with visual_chain._visual_chain_lock:
        bus = AsyncMock()
        result = await visual_chain.run_visual_chain_once(bus)

    assert result is None
    bus.rpc_request.assert_not_called()


@pytest.mark.asyncio
async def test_continuity_flows_into_the_next_run(tmp_path, monkeypatch):
    """Design doc §9 acceptance check: the *next* step's outbound context
    demonstrably contains the previous step's description -- same-run
    evidence, not schema presence. This is the exact failure mode the text
    chain's dead next_focus/drift fields represented (module docstring)."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)

    # A tiny fake "DB": load reads back whatever the last persisted chain wrote.
    db: dict[str, Any] = {"prior_description": None, "continuity_streak": 0, "context_slot_rotation": 0}
    monkeypatch.setattr(
        visual_chain,
        "load_latest_visual_chain_continuity_state",
        lambda: (db["prior_description"], db["continuity_streak"], db["context_slot_rotation"]),
    )

    def fake_persist_chain(chain):
        db["prior_description"] = chain.prior_description
        db["continuity_streak"] = chain.chain_json.get("continuity_streak", 0)
        db["context_slot_rotation"] = chain.chain_json.get("context_slot_rotation", 0)
        return True

    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", fake_persist_chain)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    generate_prompts: list[str] = []

    def fake_generate(prompt, *, base_url, timeout_sec):
        generate_prompts.append(prompt)
        return _fake_png()

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "c" * 64)

    bus = _fake_bus(_vision_result_payload("a fox curled by the fire"))
    first = await visual_chain.run_visual_chain_once(bus)
    assert first.prior_description == "a fox curled by the fire"

    bus2 = _fake_bus(_vision_result_payload("the fire has died down"))
    second = await visual_chain.run_visual_chain_once(bus2)
    assert second.prior_description == "the fire has died down"

    # The real assertion: run 2's diffusion prompt was built FROM run 1's
    # persisted description, read back through the same store the worker uses.
    assert "a fox curled by the fire" in generate_prompts[1]
    assert generate_prompts[0] == visual_chain.DEFAULT_SEED_PROMPT


@pytest.mark.asyncio
async def test_continuity_resets_after_max_runs_end_to_end(tmp_path, monkeypatch):
    """The real regression, end to end: continuity must not run unbounded.
    Drive run_visual_chain_once() through a full cap+1 cycle against the
    same fake-DB harness test_continuity_flows_into_the_next_run uses, and
    assert the run AT the cap drops prior_description from its own prompt
    (not just that resolve_visual_chain_continuity says it would in
    isolation) -- same-run evidence for the actual regression Juniper hit
    (2026-08-27: identical imagery unbroken for 10+ real runs)."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain.settings, "visual_chain_continuity_max_runs", 2)
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: "context")
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)

    db: dict[str, Any] = {"prior_description": None, "continuity_streak": 0, "context_slot_rotation": 0}
    monkeypatch.setattr(
        visual_chain,
        "load_latest_visual_chain_continuity_state",
        lambda: (db["prior_description"], db["continuity_streak"], db["context_slot_rotation"]),
    )

    def fake_persist_chain(chain):
        db["prior_description"] = chain.prior_description
        db["continuity_streak"] = chain.chain_json.get("continuity_streak", 0)
        db["context_slot_rotation"] = chain.chain_json.get("context_slot_rotation", 0)
        return True

    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", fake_persist_chain)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    generate_prompts: list[str] = []

    def fake_generate(prompt, *, base_url, timeout_sec):
        generate_prompts.append(prompt)
        return _fake_png()

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "e" * 64)

    # Review finding: a real context_slot_used here would otherwise fire
    # interpret_context_for_visual against the same fake bus configured only for the
    # vision-caption RPC -- explicit stub makes the fail-open path deliberate.
    cortex_client = _FakeCortexClient(error=RuntimeError("not exercised here"))

    # Run 1: no prior yet -- seeds from context, streak stays 0 (nothing to cap).
    r1 = await visual_chain.run_visual_chain_once(
        _fake_bus(_vision_result_payload("an aqueduct")), cortex_client=cortex_client
    )
    assert r1.chain_json["continuity_streak"] == 0
    assert r1.chain_json["continuity_reset"] is False

    # Run 2: real continuity available, streak 0 < cap(2) -- continuity used, streak -> 1.
    r2 = await visual_chain.run_visual_chain_once(
        _fake_bus(_vision_result_payload("an aqueduct at dusk")), cortex_client=cortex_client
    )
    assert "an aqueduct" in generate_prompts[1]
    assert r2.chain_json["continuity_streak"] == 1
    assert r2.chain_json["continuity_reset"] is False

    # Run 3: streak 1 < cap(2) -- still allowed, streak -> 2.
    r3 = await visual_chain.run_visual_chain_once(
        _fake_bus(_vision_result_payload("the same aqueduct again")), cortex_client=cortex_client
    )
    assert "an aqueduct at dusk" in generate_prompts[2]
    assert r3.chain_json["continuity_streak"] == 2
    assert r3.chain_json["continuity_reset"] is False

    # Run 4: streak 2 >= cap(2) -- THIS run must force a reset. Its own
    # prompt must NOT contain the prior continuity text, proving the cap
    # actually changed what got generated, not just what got recorded.
    r4 = await visual_chain.run_visual_chain_once(
        _fake_bus(_vision_result_payload("something completely different")),
        cortex_client=cortex_client,
    )
    assert "the same aqueduct again" not in generate_prompts[3]
    assert r4.chain_json["continuity_streak"] == 0
    assert r4.chain_json["continuity_reset"] is True
    # The reset run still seeds from real context, not the bland fixed
    # string -- Patch 3 and Patch 4 compose, they don't undercut each other.
    assert "context" in generate_prompts[3]


@pytest.mark.asyncio
async def test_continuity_reset_survives_a_failed_generation(tmp_path, monkeypatch):
    """Review finding: a reset run whose OWN generation fails must not
    resurrect the stale pre-reset prior_description -- the persisted row
    must carry None forward (or a real context-seeded value on the NEXT
    successful run), never the exact text the reset was meant to break out
    of. Without the fix, the next tick would read streak=0 against the
    SAME stale text and grind through another full max_runs cycle before
    resetting again -- a silent defeat of this entire PR."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    # streak already AT the cap -- this run must reset.
    monkeypatch.setattr(
        visual_chain,
        "load_latest_visual_chain_continuity_state",
        lambda: (
            "the same stale aqueduct",
            visual_chain.settings.visual_chain_continuity_max_runs,
            0,
        ),
    )

    def fake_generate(prompt, *, base_url, timeout_sec):
        raise visual_chain.DiffusionGenerationError("diffusion-host /generate returned HTTP 503")

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)

    # Patch 8: an explicit cortex_client stub, not a bare AsyncMock() bus -- this test is
    # about the generation-failure path, not interpretation, and a bare AsyncMock's
    # auto-mocked bus.rpc_request/.codec chain otherwise produces an unrelated "coroutine
    # was never awaited" warning once interpret_context_for_visual also runs.
    chain = await visual_chain.run_visual_chain_once(
        AsyncMock(), cortex_client=_FakeCortexClient(error=RuntimeError("not exercised here"))
    )

    assert chain.terminal_reason == "generation_failed"
    assert chain.chain_json["continuity_reset"] is True
    # The real assertion: NOT the stale text this reset was supposed to break.
    assert chain.prior_description is None


@pytest.mark.asyncio
async def test_continuity_reset_survives_a_failed_reobservation(tmp_path, monkeypatch):
    """Same regression, the re-observation-fails path: the image itself IS
    real (generation succeeded), but captioning fails -- the reset must
    still stick rather than falling back to the stale prior_description."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain,
        "load_latest_visual_chain_continuity_state",
        lambda: (
            "the same stale aqueduct",
            visual_chain.settings.visual_chain_continuity_max_runs,
            0,
        ),
    )
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "f" * 64)

    persisted_artifacts = []
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(
        visual_chain,
        "persist_reverie_visual_artifact",
        lambda a: persisted_artifacts.append(a) or True,
    )

    # Vision-host RPC itself raises -- request_caption swallows it, returns None.
    bus = AsyncMock()
    bus.rpc_request = AsyncMock(side_effect=TimeoutError("RPC timeout"))

    chain = await visual_chain.run_visual_chain_once(bus)

    assert chain is not None
    assert chain.terminal_reason == "max_steps"  # the image itself was real
    assert chain.chain_json["continuity_reset"] is True
    # The real assertion: NOT the stale text this reset was supposed to break.
    assert chain.prior_description is None
    assert persisted_artifacts[0].description is None  # honest, not fabricated


@pytest.mark.asyncio
async def test_run_visual_chain_once_uses_self_study_text_in_prompt_and_chain_json(
    tmp_path, monkeypatch
):
    """Patch 5 acceptance check, preserved under Patch 7's rotation: with
    self_study_text the only real context-seed available, rotation
    trivially selects it -- same discipline as the context_text test
    above."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain,
        "load_latest_self_study_reflection",
        lambda **kw: "vision events dropped 0.36x vs baseline",
    )
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "g" * 64)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    bus = _fake_bus(_vision_result_payload("a rendering of that observation"))
    chain = await visual_chain.run_visual_chain_once(
        bus, cortex_client=_FakeCortexClient(error=RuntimeError("not exercised here"))
    )

    assert chain is not None
    assert "vision events dropped 0.36x vs baseline" in chain.chain_json["prompt"]
    assert chain.chain_json["self_study_text"] == "vision events dropped 0.36x vs baseline"
    assert chain.chain_json["context_slot_used"] == "self_study"


@pytest.mark.asyncio
async def test_run_visual_chain_once_generation_failure_records_self_study_text(
    tmp_path, monkeypatch
):
    """The same self_study_text traceability holds on the generation_failed
    path -- a failed run's chain_json must still show what would have
    seeded it."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain,
        "load_latest_self_study_reflection",
        lambda **kw: "vision events dropped 0.36x vs baseline",
    )
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)

    def fake_generate(prompt, *, base_url, timeout_sec):
        raise visual_chain.DiffusionGenerationError("diffusion-host /generate returned HTTP 503")

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)

    # Patch 8: an explicit cortex_client stub, not a bare AsyncMock() bus -- this test is
    # about the generation-failure path, not interpretation, and a bare AsyncMock's
    # auto-mocked bus.rpc_request/.codec chain otherwise produces an unrelated "coroutine
    # was never awaited" warning once interpret_context_for_visual also runs.
    chain = await visual_chain.run_visual_chain_once(
        AsyncMock(), cortex_client=_FakeCortexClient(error=RuntimeError("not exercised here"))
    )

    assert chain.terminal_reason == "generation_failed"
    assert chain.chain_json["self_study_text"] == "vision events dropped 0.36x vs baseline"
    assert chain.chain_json["context_slot_used"] == "self_study"


@pytest.mark.asyncio
async def test_run_visual_chain_once_uses_memory_text_in_prompt_and_chain_json(
    tmp_path, monkeypatch
):
    """Patch 6 acceptance check, preserved under Patch 7's rotation: with
    memory_text the only real context-seed available, rotation trivially
    selects it -- same discipline as context_text/self_study_text above."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain,
        "load_latest_memory_crystallization",
        lambda **kw: "Orion and Juniper talked through the mesh work",
    )
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "h" * 64)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    bus = _fake_bus(_vision_result_payload("a rendering of that memory"))
    chain = await visual_chain.run_visual_chain_once(
        bus, cortex_client=_FakeCortexClient(error=RuntimeError("not exercised here"))
    )

    assert chain is not None
    assert "Orion and Juniper talked through the mesh work" in chain.chain_json["prompt"]
    assert chain.chain_json["memory_text"] == "Orion and Juniper talked through the mesh work"
    assert chain.chain_json["context_slot_used"] == "memory"


@pytest.mark.asyncio
async def test_run_visual_chain_once_generation_failure_records_memory_text(
    tmp_path, monkeypatch
):
    """The same memory_text traceability holds on the generation_failed
    path -- a failed run's chain_json must still show what would have
    seeded it."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain,
        "load_latest_memory_crystallization",
        lambda **kw: "Orion and Juniper talked through the mesh work",
    )

    def fake_generate(prompt, *, base_url, timeout_sec):
        raise visual_chain.DiffusionGenerationError("diffusion-host /generate returned HTTP 503")

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)

    # Patch 8: an explicit cortex_client stub, not a bare AsyncMock() bus -- this test is
    # about the generation-failure path, not interpretation, and a bare AsyncMock's
    # auto-mocked bus.rpc_request/.codec chain otherwise produces an unrelated "coroutine
    # was never awaited" warning once interpret_context_for_visual also runs.
    chain = await visual_chain.run_visual_chain_once(
        AsyncMock(), cortex_client=_FakeCortexClient(error=RuntimeError("not exercised here"))
    )

    assert chain.terminal_reason == "generation_failed"
    assert chain.chain_json["memory_text"] == "Orion and Juniper talked through the mesh work"
    assert chain.chain_json["context_slot_used"] == "memory"


@pytest.mark.asyncio
async def test_run_visual_chain_once_uses_only_one_context_seed_per_run_when_all_three_present(
    tmp_path, monkeypatch
):
    """Patch 7's actual regression fix, end to end (live report,
    2026-08-28: "the memory got washed out and Orion just continued
    generating stars"). When all three context-seeds have real content
    SIMULTANEOUSLY, only the rotation-selected one may appear in the
    prompt -- concatenating all three (Patches 3/5/6's original design) is
    exactly what silently exceeded the diffusion model's real 77-token
    budget (verified live: a real prompt hit 191 tokens)."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(
        visual_chain, "load_latest_reverie_interpretation", lambda **kw: "the coalition narration"
    )
    monkeypatch.setattr(
        visual_chain, "load_latest_self_study_reflection", lambda **kw: "the self-study finding"
    )
    monkeypatch.setattr(
        visual_chain, "load_latest_memory_crystallization", lambda **kw: "the shared memory"
    )
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "i" * 64)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    bus = _fake_bus(_vision_result_payload("a rendering"))
    chain = await visual_chain.run_visual_chain_once(
        bus, cortex_client=_FakeCortexClient(error=RuntimeError("not exercised here"))
    )

    assert chain is not None
    # Rotation starts at index 0 -> "context" wins this run.
    assert chain.chain_json["context_slot_used"] == "context"
    assert "the coalition narration" in chain.chain_json["prompt"]
    # The real assertion: the other two, though real and separately
    # recorded, must NOT also be crammed into the same prompt string.
    assert "the self-study finding" not in chain.chain_json["prompt"]
    assert "the shared memory" not in chain.chain_json["prompt"]
    # Still recorded, for inspectability -- just not rendered this run.
    assert chain.chain_json["self_study_text"] == "the self-study finding"
    assert chain.chain_json["memory_text"] == "the shared memory"


@pytest.mark.asyncio
async def test_context_slot_rotation_advances_across_successive_runs(tmp_path, monkeypatch):
    """Design doc §18 acceptance check: successive runs visit different
    context-seeds in turn, proven via the same fake-DB round-trip harness
    test_continuity_flows_into_the_next_run uses for prior_description --
    same-run evidence, not just select_context_slot's own isolated
    correctness."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_reverie_interpretation", lambda **kw: "the coalition narration"
    )
    monkeypatch.setattr(
        visual_chain, "load_latest_self_study_reflection", lambda **kw: "the self-study finding"
    )
    monkeypatch.setattr(
        visual_chain, "load_latest_memory_crystallization", lambda **kw: "the shared memory"
    )

    db: dict[str, Any] = {"prior_description": None, "continuity_streak": 0, "context_slot_rotation": 0}
    monkeypatch.setattr(
        visual_chain,
        "load_latest_visual_chain_continuity_state",
        lambda: (db["prior_description"], db["continuity_streak"], db["context_slot_rotation"]),
    )

    def fake_persist_chain(chain):
        db["prior_description"] = chain.prior_description
        db["continuity_streak"] = chain.chain_json.get("continuity_streak", 0)
        db["context_slot_rotation"] = chain.chain_json.get("context_slot_rotation", 0)
        return True

    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", fake_persist_chain)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "j" * 64)

    cortex_client = _FakeCortexClient(error=RuntimeError("not exercised here"))
    slots_used = []
    for _ in range(4):
        chain = await visual_chain.run_visual_chain_once(
            _fake_bus(_vision_result_payload("a rendering")), cortex_client=cortex_client
        )
        slots_used.append(chain.chain_json["context_slot_used"])

    assert slots_used == ["context", "self_study", "memory", "context"]


# --- Patch 8: metacog interpretation step -----------------------------------
#
# The real remaining gap after Patch 7 fixed which clause reaches the model:
# build_visual_prompt is pure string concatenation, so the diffusion model
# pattern-matches whatever concrete nouns happen to be in the raw text --
# generic abstract prose has none, hence the clouds/nebulas/aqueducts
# fallback. interpret_context_for_visual is the fix: one metacog-routed
# cortex-exec call that invents a concrete visual metaphor before the prompt
# is built. Fails open to the raw slot text on ANY failure -- Patch 7's
# behavior is the fallback, not replaced.


class _FakeCortexClient:
    """Stub CortexExecClient -- captures the plan_request it was given and
    returns/raises whatever the test configures, no real bus/RPC plumbing."""

    def __init__(self, *, result: dict[str, Any] | None = None, error: BaseException | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def execute_plan(self, *, source, req, correlation_id, timeout_sec):
        self.calls.append(
            {"source": source, "req": req, "correlation_id": correlation_id, "timeout_sec": timeout_sec}
        )
        if self.error is not None:
            raise self.error
        return self.result


def test_build_visual_interpretation_plan_request_always_plain_metacog():
    """Never metacog_background -- this call must always dispatch immediately,
    per Juniper's stated priority ('diffusion trumps text if we are too
    tight'). Unlike reverie.py's _metacog_route(), there is no flag branch
    here at all -- this is load-bearing, not an oversight."""
    from app import visual_chain

    req = visual_chain.build_visual_interpretation_plan_request(
        source_label="Orion remembers",
        source_text="we moved a server between Ethernet ports",
        prior_description="a lantern swinging over black water",
        correlation_id="corr-1",
    )
    assert req.args.extra["llm_route"] == "metacog"
    assert req.args.extra["mode"] == "metacog"
    assert req.context["options"]["llm_lane"] == "background"
    assert req.context["options"]["allow_chat_fallback"] is False
    assert req.context["source_label"] == "Orion remembers"
    assert req.context["source_text"] == "we moved a server between Ethernet ports"
    assert req.context["prior_description"] == "a lantern swinging over black water"


def test_build_visual_interpretation_plan_request_blank_prior_becomes_none():
    from app import visual_chain

    req = visual_chain.build_visual_interpretation_plan_request(
        source_label="Orion remembers", source_text="x", prior_description="   ",
        correlation_id="corr-1",
    )
    assert req.context["prior_description"] is None


@pytest.mark.asyncio
async def test_interpret_context_for_visual_returns_stripped_text_on_success():
    from app import visual_chain

    client = _FakeCortexClient(result={"final_text": "  a lantern over black water  "})
    text = await visual_chain.interpret_context_for_visual(
        AsyncMock(), cortex_client=client, slot_name="memory", slot_text="raw clause",
        prior_description=None, correlation_id="corr-1", timeout_sec=30.0,
    )
    assert text == "a lantern over black water"
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_interpret_context_for_visual_none_on_timeout():
    import asyncio as _asyncio

    from app import visual_chain

    client = _FakeCortexClient(error=_asyncio.TimeoutError())
    text = await visual_chain.interpret_context_for_visual(
        AsyncMock(), cortex_client=client, slot_name="context", slot_text="raw clause",
        prior_description=None, correlation_id="corr-1", timeout_sec=30.0,
    )
    assert text is None


@pytest.mark.asyncio
async def test_interpret_context_for_visual_none_on_malformed_result():
    from app import visual_chain

    client = _FakeCortexClient(result={"no_final_text_or_steps_here": True})
    text = await visual_chain.interpret_context_for_visual(
        AsyncMock(), cortex_client=client, slot_name="context", slot_text="raw clause",
        prior_description=None, correlation_id="corr-1", timeout_sec=30.0,
    )
    assert text is None


@pytest.mark.asyncio
async def test_interpret_context_for_visual_none_on_blank_text():
    from app import visual_chain

    client = _FakeCortexClient(result={"final_text": "   "})
    text = await visual_chain.interpret_context_for_visual(
        AsyncMock(), cortex_client=client, slot_name="context", slot_text="raw clause",
        prior_description=None, correlation_id="corr-1", timeout_sec=30.0,
    )
    assert text is None


@pytest.mark.asyncio
async def test_run_visual_chain_once_uses_interpreted_text_in_prompt(tmp_path, monkeypatch):
    """The actual fix, end to end: when interpretation succeeds, the concrete
    metaphor -- not the raw abstract clause -- is what reaches the diffusion
    prompt, and both are recorded (chain_json keeps the raw slot text under
    its own field unchanged; the interpretation is a new, separate field)."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain,
        "load_latest_memory_crystallization",
        lambda **kw: "we moved a server between Ethernet ports",
    )

    generate_calls: list[str] = []
    monkeypatch.setattr(
        visual_chain, "call_diffusion_generate",
        lambda prompt, **kw: generate_calls.append(prompt) or _fake_png(),
    )
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "e" * 64)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    client = _FakeCortexClient(
        result={"final_text": "a tangle of unplugged Ethernet cables coiled beside an empty rack slot"}
    )
    bus = _fake_bus(_vision_result_payload("a rendering"))
    chain = await visual_chain.run_visual_chain_once(bus, cortex_client=client)

    assert chain is not None
    assert len(client.calls) == 1
    assert "unplugged Ethernet cables" in generate_calls[0]
    assert "we moved a server between Ethernet ports" not in generate_calls[0]
    assert chain.chain_json["memory_text"] == "we moved a server between Ethernet ports"
    assert (
        chain.chain_json["context_slot_interpreted"]
        == "a tangle of unplugged Ethernet cables coiled beside an empty rack slot"
    )


@pytest.mark.asyncio
async def test_run_visual_chain_once_falls_back_to_raw_slot_text_on_interpretation_failure(
    tmp_path, monkeypatch
):
    """Fail-open, exactly Patch 7's own behavior when nothing was selected --
    a metacog outage must degrade to the raw clause, never break the run."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain, "load_latest_memory_crystallization", lambda **kw: "the raw clause"
    )

    generate_calls: list[str] = []
    monkeypatch.setattr(
        visual_chain, "call_diffusion_generate",
        lambda prompt, **kw: generate_calls.append(prompt) or _fake_png(),
    )
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "f" * 64)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    client = _FakeCortexClient(error=RuntimeError("metacog lane down"))
    bus = _fake_bus(_vision_result_payload("a rendering"))
    chain = await visual_chain.run_visual_chain_once(bus, cortex_client=client)

    assert chain is not None
    assert "the raw clause" in generate_calls[0]
    assert chain.chain_json["context_slot_interpreted"] is None


@pytest.mark.asyncio
async def test_run_visual_chain_once_skips_interpretation_when_disabled(tmp_path, monkeypatch):
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain.settings, "visual_chain_interpretation_enabled", False)
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain, "load_latest_memory_crystallization", lambda **kw: "the raw clause"
    )
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "g" * 64)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    client = _FakeCortexClient(result={"final_text": "should never be called"})
    bus = _fake_bus(_vision_result_payload("a rendering"))
    chain = await visual_chain.run_visual_chain_once(bus, cortex_client=client)

    assert chain is not None
    assert client.calls == []
    assert chain.chain_json["context_slot_interpreted"] is None
    assert "the raw clause" in chain.chain_json["prompt"]


@pytest.mark.asyncio
async def test_run_visual_chain_once_skips_interpretation_when_no_slot_available(
    tmp_path, monkeypatch
):
    """Nothing to interpret when select_context_slot itself picked nothing --
    the interpretation call must not fire on an empty/None slot."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_continuity_state", lambda: (None, 0, 0)
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "h" * 64)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    client = _FakeCortexClient(result={"final_text": "should never be called"})
    bus = _fake_bus(_vision_result_payload("a rendering"))
    chain = await visual_chain.run_visual_chain_once(bus, cortex_client=client)

    assert chain is not None
    assert client.calls == []
    assert chain.chain_json["context_slot_interpreted"] is None
