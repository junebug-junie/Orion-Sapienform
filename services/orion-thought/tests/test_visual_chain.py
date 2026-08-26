"""Reverie VISUAL chain (Patch 2) -- orchestration is fully testable with
injected fakes at each hop: diffusion generation, percept upload, vision-host
RPC, and persistence. No GPU, no torch, no live Redis/Postgres.
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


def test_build_visual_prompt_uses_seed_when_no_prior():
    from app import visual_chain

    assert visual_chain.build_visual_prompt(None) == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt("   ") == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt(None, None) == visual_chain.DEFAULT_SEED_PROMPT
    assert visual_chain.build_visual_prompt("  ", "  ") == visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_continues_prior():
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt("a quiet room, warm light")
    assert "a quiet room, warm light" in prompt
    assert prompt != visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_mesh_context_only_when_no_prior():
    """First-ever run (or a run whose own continuity read failed) with a real
    mesh_context available -- uses it directly rather than falling back to
    the placeholder DEFAULT_SEED_PROMPT."""
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt(None, "the mesh is quiet tonight, low traffic")
    assert "the mesh is quiet tonight, low traffic" in prompt
    assert prompt != visual_chain.DEFAULT_SEED_PROMPT


def test_build_visual_prompt_weaves_both_prior_and_mesh_context():
    """The real breaking-the-self-loop case: both inputs present, both must
    actually appear in the resulting prompt (not one silently dropped)."""
    from app import visual_chain

    prompt = visual_chain.build_visual_prompt(
        "a fox curled by the fire", "open loop: the deploy queue is backed up"
    )
    assert "a fox curled by the fire" in prompt
    assert "open loop: the deploy queue is backed up" in prompt


def test_truncate_mesh_context_bounds_length_and_normalizes_blank():
    from app import visual_chain

    # Real prose with word boundaries -- a hard character slice would cut
    # mid-word (review finding); truncate_at_word_boundary breaks on
    # whitespace instead, so the result must not exceed the limit but also
    # must not end mid-word.
    long_text = "the mesh is busy right now " * 40  # far past the 400-char default
    truncated = visual_chain._truncate_mesh_context(long_text, max_chars=50)
    assert truncated is not None
    assert len(truncated) <= 51  # <= limit + the "…" truncation marker
    # Delegates to the shared word-boundary helper (existing-mechanism check)
    # -- confirm it's really that function's behavior, not a coincidence:
    # every word in the source text is one of these five, so an ending on
    # any of them (not a mid-word fragment) proves the boundary held.
    stripped = truncated.rstrip("…").rstrip()
    assert stripped.split()[-1] in {"the", "mesh", "is", "busy", "right", "now"}

    assert visual_chain._truncate_mesh_context(None) is None
    assert visual_chain._truncate_mesh_context("   ") is None
    assert visual_chain._truncate_mesh_context("") is None


def test_prompt_source_flags_reflect_what_was_actually_passed():
    from app import visual_chain

    assert visual_chain._prompt_source_flags(None, None) == {
        "used_prior": False,
        "used_mesh": False,
    }
    assert visual_chain._prompt_source_flags("  ", None) == {
        "used_prior": False,
        "used_mesh": False,
    }
    assert visual_chain._prompt_source_flags("a fox by the fire", None) == {
        "used_prior": True,
        "used_mesh": False,
    }
    assert visual_chain._prompt_source_flags(None, "mesh signal") == {
        "used_prior": False,
        "used_mesh": True,
    }
    assert visual_chain._prompt_source_flags("prior", "mesh") == {
        "used_prior": True,
        "used_mesh": True,
    }


def test_truncate_mesh_context_uses_the_settings_char_limit_by_default(monkeypatch):
    """No module-level constant anymore (review finding: every other tunable
    in this file is a settings.py field with an ORION_ env alias) -- confirm
    the default actually comes from settings, not a bare magic number."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_mesh_context_char_limit", 10)
    truncated = visual_chain._truncate_mesh_context("this text is much longer than ten chars")
    assert truncated is not None
    assert len(truncated) <= 11


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
    monkeypatch.setattr(
        visual_chain,
        "load_recent_reverie_interpretation",
        lambda **kw: "the substrate is quiet, one open loop about a stalled deploy",
    )

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
    # No prior_description, but a real mesh_context WAS available -- the
    # prompt must be built from it, not silently fall back to the seed.
    assert generate_calls != [visual_chain.DEFAULT_SEED_PROMPT]
    assert "stalled deploy" in generate_calls[0]
    # And the exact mesh_context used must be persisted alongside the prompt
    # (cockpit "what's influencing reverie" -- must match what was embedded).
    assert (
        persisted_chains[0].chain_json["mesh_context"]
        == "the substrate is quiet, one open loop about a stalled deploy"
    )
    # Ground-truth prompt-source flags (review finding, replaces the cockpit
    # guessing from prompt text): no prior_description this run, but mesh
    # context WAS used.
    assert persisted_chains[0].chain_json["used_prior"] is False
    assert persisted_chains[0].chain_json["used_mesh"] is True

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
async def test_run_visual_chain_once_neither_prior_nor_mesh_uses_seed_and_records_both_flags_false(
    tmp_path, monkeypatch
):
    """Review finding: the true first-ever-run branch (both reads empty,
    DEFAULT_SEED_PROMPT used) was only unit-tested for _prompt_source_flags
    in isolation -- nothing exercised it end-to-end through
    run_visual_chain_once to confirm the persisted chain_json actually says
    used_prior=False AND used_mesh=False together, not just individually."""
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_visual_chain_prior_description", lambda: None)
    monkeypatch.setattr(visual_chain, "load_recent_reverie_interpretation", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda prompt, **kw: _fake_png())
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda data, **kw: "d" * 64)

    persisted_chains = []
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_chain", lambda c: persisted_chains.append(c) or True
    )
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_artifact", lambda a: True)

    bus = _fake_bus(_vision_result_payload("something new"))
    chain = await visual_chain.run_visual_chain_once(bus)

    assert chain is not None
    assert len(persisted_chains) == 1
    cj = persisted_chains[0].chain_json
    assert cj["prompt"] == visual_chain.DEFAULT_SEED_PROMPT
    assert cj["used_prior"] is False
    assert cj["used_mesh"] is False
    assert cj["mesh_context"] is None


@pytest.mark.asyncio
async def test_run_visual_chain_once_generation_failure_writes_no_artifact(tmp_path, monkeypatch):
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_prior_description", lambda: "old description"
    )
    monkeypatch.setattr(visual_chain, "load_recent_reverie_interpretation", lambda **kw: "mesh signal")

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
    # mesh_context was resolved and embedded in the (failed) attempted
    # prompt -- it must still be visible in the failure's chain_json so the
    # cockpit can show what was going to influence this run.
    assert persisted_chains[0].chain_json["mesh_context"] == "mesh signal"
    assert "mesh signal" in persisted_chains[0].chain_json["prompt"]
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
    monkeypatch.setattr(visual_chain, "load_recent_reverie_interpretation", lambda **kw: None)
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

    # A tiny fake "DB": load reads back whatever the last persisted chain wrote.
    db: dict[str, str | None] = {"prior_description": None}
    monkeypatch.setattr(
        visual_chain, "load_latest_visual_chain_prior_description", lambda: db["prior_description"]
    )
    monkeypatch.setattr(visual_chain, "load_recent_reverie_interpretation", lambda **kw: None)

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
