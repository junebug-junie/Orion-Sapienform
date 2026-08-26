"""Eval: reverie VISUAL chain honest-degradation invariants.

Distinct from `tests/test_visual_chain.py`'s unit tests (each of which checks
one hop's failure in isolation): this scores a small matrix of hop-failure
combinations against the invariants `app/visual_chain.py`'s module docstring
promises --

  - a real image was generated and stored -> terminal_reason="max_steps",
    even if re-observation (percept upload / vision-host RPC) failed.
  - nothing was generated -> terminal_reason="generation_failed", and
    prior_description must NOT change (nothing here to advance continuity
    with).
  - `description` is either a real, non-empty string or `None` -- NEVER an
    empty string, a placeholder, or fabricated text standing in for a
    caption that didn't arrive. This is the §0A "no empty-shell cognition"
    bar applied to this specific chain, and it is the one property none of
    the individual unit tests assert across every failure combination at
    once.

Run: pytest services/orion-thought/evals -q
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _fake_png() -> bytes:
    return PNG_SIG + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 32, 32)


class _FakeCodec:
    def decode(self, data):
        return SimpleNamespace(ok=True, error=None, envelope=SimpleNamespace(payload=data))


def _vision_payload(caption: str) -> dict:
    """`caption` here is always a string (possibly empty) -- the scenario
    where vision-host never replies at all is exercised separately via
    `bus.rpc_request`'s own side_effect, not through this payload builder."""
    return {
        "ok": True,
        "task_type": "caption_frame",
        "artifact": {
            "artifact_id": "a", "correlation_id": "c", "task_type": "caption_frame",
            "device": "cuda:0", "inputs": {}, "timing": {}, "model_fingerprints": {},
            "outputs": {"caption": {"text": caption, "confidence": 1.0}},
        },
    }


@dataclass
class Scenario:
    name: str
    generate_ok: bool
    store_ok: bool
    upload_ok: bool
    caption: str | None  # None = RPC/decode fails; "" = vision-host returns an empty caption
    expected_terminal: str
    expected_description_is_real: bool  # True: non-empty str; False: must be None


SCENARIOS = [
    Scenario("generate_fails", False, True, True, "a room", "generation_failed", False),
    Scenario("store_fails", True, False, True, "a room", "generation_failed", False),
    Scenario("upload_fails", True, True, False, "a room", "max_steps", False),
    Scenario("caption_rpc_fails", True, True, True, None, "max_steps", False),
    Scenario("caption_empty", True, True, True, "", "max_steps", False),
    Scenario("full_success", True, True, True, "a quiet room, warm light", "max_steps", True),
]


async def _run_scenario(monkeypatch, scenario: Scenario, tmp_path):
    from app import visual_chain

    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_visual_chain_prior_description", lambda: "old")
    # Mesh-context seeding (2026-08-26): mocked here too, not left to hit the
    # real (unmocked) DB engine -- review finding: this eval was updated for
    # the new load_recent_reverie_interpretation dependency, but the honesty
    # invariants it was written to police were not yet extended to cover the
    # new mesh_context/used_prior/used_mesh chain_json fields until this diff.
    monkeypatch.setattr(
        visual_chain, "load_recent_reverie_interpretation", lambda **kw: "mesh signal for this scenario"
    )

    def fake_generate(prompt, **kw):
        if not scenario.generate_ok:
            raise visual_chain.DiffusionGenerationError("boom")
        return _fake_png()

    def fake_store(data, **kw):
        if not scenario.store_ok:
            raise RuntimeError("disk full")
        from orion.reverie.visual_storage import store_visual_artifact

        return store_visual_artifact(data, base_dir=str(tmp_path))

    def fake_upload(data, **kw):
        if not scenario.upload_ok:
            raise visual_chain.PerceptUploadError("upload boom")
        return "a" * 64

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "store_visual_artifact", fake_store)
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", fake_upload)

    persisted_artifacts = []
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_artifact", lambda a: persisted_artifacts.append(a) or True
    )

    bus = AsyncMock()
    if scenario.caption is None:
        bus.rpc_request = AsyncMock(side_effect=TimeoutError("rpc timeout"))
    else:
        bus.codec = _FakeCodec()
        bus.rpc_request = AsyncMock(return_value={"data": _vision_payload(scenario.caption)})

    chain = await visual_chain.run_visual_chain_once(bus)
    return chain, persisted_artifacts


def _check(chain, persisted_artifacts, scenario: Scenario) -> list[str]:
    problems = []
    if chain is None:
        return [f"{scenario.name}: run_visual_chain_once returned None unexpectedly"]
    if chain.terminal_reason != scenario.expected_terminal:
        problems.append(
            f"{scenario.name}: terminal_reason={chain.terminal_reason!r}, "
            f"expected {scenario.expected_terminal!r}"
        )

    # Mesh-context honesty (2026-08-26): every scenario here has both a real
    # prior_description ("old") and a real mesh_context mocked in -- the
    # persisted chain_json must say so truthfully in every scenario,
    # including the failure ones (an operator inspecting a failed run still
    # needs to see what was going to influence it -- module docstring's
    # inspectable-evidence bar applies to failures too, not just successes).
    cj = chain.chain_json
    if cj.get("mesh_context") != "mesh signal for this scenario":
        problems.append(f"{scenario.name}: mesh_context not persisted as-given: {cj.get('mesh_context')!r}")
    if cj.get("used_prior") is not True:
        problems.append(f"{scenario.name}: used_prior flag wrong: {cj.get('used_prior')!r} (prior was real)")
    if cj.get("used_mesh") is not True:
        problems.append(f"{scenario.name}: used_mesh flag wrong: {cj.get('used_mesh')!r} (mesh was real)")

    if scenario.expected_terminal == "generation_failed":
        if chain.prior_description != "old":
            problems.append(
                f"{scenario.name}: prior_description changed on generation_failed "
                f"({chain.prior_description!r})"
            )
        if persisted_artifacts:
            problems.append(f"{scenario.name}: an artifact was persisted despite no image")
        return problems

    # max_steps: description must be a real non-empty string or exactly None.
    description = persisted_artifacts[0].description if persisted_artifacts else None
    if description == "":
        problems.append(f"{scenario.name}: description is an empty string, not None (fabrication risk)")
    if scenario.expected_description_is_real:
        if not description:
            problems.append(f"{scenario.name}: expected a real description, got {description!r}")
        if chain.prior_description != description:
            problems.append(f"{scenario.name}: prior_description did not advance to the new description")
    else:
        if description is not None:
            problems.append(f"{scenario.name}: expected no description, got {description!r}")
        if chain.prior_description != "old":
            problems.append(
                f"{scenario.name}: prior_description was not carried forward on a failed "
                f"re-observation ({chain.prior_description!r})"
            )
    return problems


@pytest.mark.asyncio
async def test_visual_chain_honest_degradation_matrix(monkeypatch, tmp_path):
    all_problems: list[str] = []
    for scenario in SCENARIOS:
        with monkeypatch.context() as m:
            chain, persisted_artifacts = await _run_scenario(m, scenario, tmp_path / scenario.name)
            all_problems.extend(_check(chain, persisted_artifacts, scenario))
    assert not all_problems, "honest-degradation invariant violated:\n" + "\n".join(all_problems)
