"""Reverie VISUAL chain (Patch 2 orchestration + Patch 3 context-seeding) --
fully testable with injected fakes at each hop: diffusion generation, percept
upload, vision-host RPC, reverie-interpretation context-seed, and
persistence. No GPU, no torch, no live Redis/Postgres.
"""
from __future__ import annotations

import struct
from types import SimpleNamespace
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


# --- build_visual_prompt (pure) ------------------------------------------------


def test_build_visual_prompt_uses_seed_when_neither_prior_nor_context():
    from app import visual_chain

    assert visual_chain.build_visual_prompt(None) == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt("   ") == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt(None, "   ") == visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_continues_prior():
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt("a quiet room, warm light")
    assert "a quiet room, warm light" in prompt
    assert prompt != visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_uses_context_when_no_prior():
    """Patch 3: a cold-start run (no prior_description yet) with a real
    reverie thought on record must seed from that thought, not the generic
    fixed string -- the whole point of context-seeding."""
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt(None, "the mesh keeps humming with new nodes")
    assert "the mesh keeps humming with new nodes" in prompt
    assert prompt != visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_blends_prior_and_context():
    """Both continuity and context-seed present -- neither silently drops
    the other; a reader of the stored prompt can see both inputs."""
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt("a quiet room, warm light", "curiosity about the mesh")
    assert "a quiet room, warm light" in prompt
    assert "curiosity about the mesh" in prompt


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
    monkeypatch.setattr(visual_chain, "load_latest_visual_chain_prior_description", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)

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
        visual_chain, "load_latest_visual_chain_prior_description", lambda: "old description"
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)

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
        visual_chain, "load_latest_visual_chain_prior_description", lambda: "old description"
    )
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: None)
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
    """Patch 3 acceptance check: the context-seed actually reaches the
    diffusion prompt AND is recorded in chain_json as its own field -- same-
    run evidence a reader (or this repo's own Hub Reverie tab) can inspect,
    not just embedded prose inside the prompt string (module docstring's
    "same-run evidence, not schema presence" discipline, design doc §9)."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_visual_chain_prior_description", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain, "load_latest_reverie_interpretation", lambda **kw: "a real reverie thought"
    )
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "d" * 64)

    persisted_chains = []
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_chain", lambda c: persisted_chains.append(c) or True
    )
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    bus = _fake_bus(_vision_result_payload("a rendering of that thought"))
    chain = await visual_chain.run_visual_chain_once(bus)

    assert chain is not None
    assert "a real reverie thought" in chain.chain_json["prompt"]
    assert chain.chain_json["context_text"] == "a real reverie thought"


@pytest.mark.asyncio
async def test_run_visual_chain_once_generation_failure_records_context_text(
    tmp_path, monkeypatch
):
    """The same context_text traceability holds on the generation_failed path
    -- a failed run's chain_json must still show what would have seeded it."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_visual_chain_prior_description", lambda **kw: None)
    monkeypatch.setattr(
        visual_chain, "load_latest_reverie_interpretation", lambda **kw: "a real reverie thought"
    )

    def fake_generate(prompt, *, base_url, timeout_sec):
        raise visual_chain.DiffusionGenerationError("diffusion-host /generate returned HTTP 503")

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)

    chain = await visual_chain.run_visual_chain_once(AsyncMock())

    assert chain.terminal_reason == "generation_failed"
    assert chain.chain_json["context_text"] == "a real reverie thought"


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

    # A tiny fake "DB": load reads back whatever the last persisted chain wrote.
    db: dict[str, str | None] = {"prior_description": None}
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_prior_description", lambda: db["prior_description"]
    )

    def fake_persist_chain(chain):
        db["prior_description"] = chain.prior_description
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
