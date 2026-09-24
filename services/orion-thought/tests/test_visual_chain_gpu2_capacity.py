"""GPU2 capacity mutex on the diffusion call site.

orion-world-model shares circe's GPU2 with orion-diffusion-host, with no
OS-level arbitration (live-confirmed 2026-09-24: two back-to-back "CUDA
error: CUDA-capable device(s) is/are busy or unavailable" failures). These
tests assert the properties that actually prevent that recurring: capacity
is acquired before any GPU call, a rejection never reaches the GPU and is
recorded as a normal resource_deferred (not a generation failure), and the
permit is always released -- success or failure -- so a stuck permit can
never starve the other side of the mutex.
"""
from __future__ import annotations

import asyncio

import pytest

from app import visual_chain


def _persist_spy(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_chain", lambda c: persisted.append(c) or True
    )
    return persisted


def _no_thermal_refusal(monkeypatch):
    async def _allow():
        return visual_chain.ThermalVerdict(
            state="normal", temp_c=20.0, age_sec=1.0, allows_gpu_work=True, reason="ok"
        )
    monkeypatch.setattr(visual_chain, "evaluate_thermal_gate", _allow)


class FakePermit:
    """Records acquire/close calls; `acquire_error` lets a test script a
    rejection without a real capacity authority."""

    instances: list["FakePermit"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.acquired = False
        self.closed = False
        self.acquire_error: Exception | None = None
        FakePermit.instances.append(self)

    async def acquire(self) -> "FakePermit":
        if self.acquire_error is not None:
            raise self.acquire_error
        self.acquired = True
        return self

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    FakePermit.instances.clear()
    _no_thermal_refusal(monkeypatch)
    monkeypatch.setattr(visual_chain.settings, "visual_chain_gpu2_capacity_enabled", True)


def test_disabled_flag_never_constructs_a_permit(monkeypatch):
    monkeypatch.setattr(visual_chain.settings, "visual_chain_gpu2_capacity_enabled", False)
    called = []
    monkeypatch.setattr(visual_chain, "GpuCapacityPermit", lambda **k: called.append(k) or FakePermit(**k))
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", lambda *a, **k: b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    _persist_spy(monkeypatch)

    asyncio.run(visual_chain.run_visual_chain_once(bus=None))

    assert called == [], "capacity permit must not be constructed when the flag is off"


def test_capacity_rejection_never_reaches_the_gpu_and_records_resource_deferred(monkeypatch):
    def _make(**kwargs):
        permit = FakePermit(**kwargs)
        permit.acquire_error = visual_chain.CapacityRejected("capacity_wait_budget_exhausted")
        return permit
    monkeypatch.setattr(visual_chain, "GpuCapacityPermit", _make)

    def _boom(*a, **k):
        raise AssertionError("call_diffusion_generate must not run when capacity was rejected")
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", _boom)
    persisted = _persist_spy(monkeypatch)

    chain = asyncio.run(visual_chain.run_visual_chain_once(bus=None))

    assert chain is not None
    assert chain.terminal_reason == "resource_deferred"
    assert "gpu2_capacity" in chain.chain_json["resource_gate"]["reason"]
    assert persisted == [chain]


def test_permit_acquired_before_generate_and_released_on_success(monkeypatch):
    monkeypatch.setattr(visual_chain, "GpuCapacityPermit", FakePermit)
    order: list[str] = []

    def _generate(*a, **k):
        order.append("generate")
        assert FakePermit.instances[0].acquired, "permit must be acquired before the GPU call"
        assert not FakePermit.instances[0].closed, "permit must not be released before the GPU call finishes"
        return b"\x89PNG\r\n\x1a\n" + b"0" * 32

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", _generate)
    monkeypatch.setattr(
        visual_chain, "store_visual_artifact",
        lambda *a, **k: visual_chain.StoredVisualArtifact(
            sha256="y" * 64, mime="image/png", bytes=1, width=1, height=1, path="x"
        ),
    )
    monkeypatch.setattr(visual_chain, "upload_to_percept_store", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no percept store in this test")))
    monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", lambda c: True)
    monkeypatch.setattr(visual_chain, "acknowledge_visual_production", lambda *a, **k: None)

    asyncio.run(visual_chain.run_visual_chain_once(bus=None))

    assert order == ["generate"]
    assert len(FakePermit.instances) == 1
    assert FakePermit.instances[0].acquired and FakePermit.instances[0].closed


def test_permit_released_even_when_generation_fails(monkeypatch):
    monkeypatch.setattr(visual_chain, "GpuCapacityPermit", FakePermit)

    def _boom(*a, **k):
        raise visual_chain.DiffusionGenerationError("500")
    monkeypatch.setattr(visual_chain, "call_diffusion_generate", _boom)
    _persist_spy(monkeypatch)

    chain = asyncio.run(visual_chain.run_visual_chain_once(bus=None))

    assert chain.terminal_reason == "generation_failed"
    assert len(FakePermit.instances) == 1
    assert FakePermit.instances[0].acquired and FakePermit.instances[0].closed, (
        "a stuck, unreleased permit would starve the other side of the mutex "
        "(orion-world-model) even after this run gives up"
    )
