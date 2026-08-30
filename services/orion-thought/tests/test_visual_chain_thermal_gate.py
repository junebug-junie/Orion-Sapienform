"""The ambient thermal gate on the visual chain.

GPU work heats the room Juniper sits in. This is the only budget in the repo
whose referent is external to Orion, so the interesting properties are that it
actually withholds work, that it RECORDS the refusal rather than skipping
silently, and that a dead sensor does not quietly remove the capability.
"""

from __future__ import annotations

import asyncio

import pytest

from app import visual_chain


@pytest.fixture(autouse=True)
def _reset_gate_state(monkeypatch):
    monkeypatch.setattr(visual_chain, "_thermal_state", "normal", raising=False)
    monkeypatch.setattr(visual_chain.settings, "thermal_gate_enabled", True)
    monkeypatch.setattr(visual_chain.settings, "thermal_hot_c", 32.0)
    monkeypatch.setattr(visual_chain.settings, "thermal_hot_rearm_c", 30.5)


def _persist_spy(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_chain", lambda c: persisted.append(c) or True
    )
    return persisted


def _no_gpu_allowed(monkeypatch):
    """Any call to the diffusion host is a test failure: the whole point is that
    the request is never made."""

    def _boom(*args, **kwargs):
        raise AssertionError("call_diffusion_generate must not run while the room is hot")

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", _boom)


class TestRefusal:
    def test_a_hot_room_withholds_the_gpu_request(self, monkeypatch) -> None:
        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (34.0, 1.0))
        _no_gpu_allowed(monkeypatch)
        persisted = _persist_spy(monkeypatch)

        chain = asyncio.run(visual_chain.run_visual_chain_once(bus=None))

        assert chain is not None
        assert chain.terminal_reason == "thermal_refused"
        assert persisted == [chain], "the refusal must be recorded, not just skipped"

    def test_the_recorded_refusal_carries_the_reading_that_caused_it(self, monkeypatch) -> None:
        """A refusal that does not say what it read cannot be argued with."""
        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (34.2, 2.0))
        _no_gpu_allowed(monkeypatch)
        _persist_spy(monkeypatch)

        chain = asyncio.run(visual_chain.run_visual_chain_once(bus=None))

        gate = chain.chain_json["thermal_gate"]
        assert gate["state"] == "hot"
        assert gate["temp_c"] == 34.2
        assert gate["hot_c"] == 32.0

    def test_refusal_returns_a_chain_not_none(self, monkeypatch) -> None:
        """None already means "a run was already in flight". Reusing it for a
        refusal would collapse two different facts into one signal."""
        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (34.0, 1.0))
        _no_gpu_allowed(monkeypatch)
        _persist_spy(monkeypatch)

        assert asyncio.run(visual_chain.run_visual_chain_once(bus=None)) is not None


class TestHysteresisAcrossRuns:
    def test_the_gate_stays_shut_between_rearm_and_trip(self, monkeypatch) -> None:
        """31.0C is under the 32.0 trip but over the 30.5 re-arm. Without
        cross-run state the gate would reopen here and the room would get a run
        it should not have."""
        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (34.0, 1.0))
        _no_gpu_allowed(monkeypatch)
        _persist_spy(monkeypatch)
        asyncio.run(visual_chain.run_visual_chain_once(bus=None))
        assert visual_chain._thermal_state == "hot"

        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (31.0, 1.0))
        chain = asyncio.run(visual_chain.run_visual_chain_once(bus=None))
        assert chain.terminal_reason == "thermal_refused"

    def test_a_failed_read_does_not_reset_the_held_state(self, monkeypatch) -> None:
        """One dropped reading must not silently drop the hysteresis and hand a
        hot room a free run on the next good sample."""
        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (34.0, 1.0))
        _no_gpu_allowed(monkeypatch)
        _persist_spy(monkeypatch)
        asyncio.run(visual_chain.run_visual_chain_once(bus=None))

        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (None, None))
        asyncio.run(visual_chain.run_visual_chain_once(bus=None))
        assert visual_chain._thermal_state == "hot"


class TestDegradedAndDisabled:
    """Asserted at the gate boundary rather than by running the whole chain: the
    fall-through path does real generation, and a test that reaches the network
    and the filesystem to prove a branch was NOT taken is both slow and lying
    about what it measures."""

    def test_a_dead_sensor_does_not_block_work(self, monkeypatch) -> None:
        """Fail-open, deliberately: the cost of wrongly allowing is a warm room
        for one cycle; the cost of wrongly blocking is a capability that
        disappears with no error anywhere."""
        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (None, None))

        verdict = visual_chain.evaluate_thermal_gate()

        assert verdict.allows_gpu_work is True
        assert verdict.degraded is True, "an allow on no reading must announce itself"
        assert verdict.state == "unknown"

    def test_a_stale_reading_does_not_latch_the_gate_shut(self, monkeypatch) -> None:
        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (40.0, 6000.0))

        verdict = visual_chain.evaluate_thermal_gate()

        assert verdict.allows_gpu_work is True
        assert verdict.degraded is True

    def test_the_gate_can_be_turned_off(self, monkeypatch) -> None:
        """The kill switch has to actually reach the decision, not just exist."""
        monkeypatch.setattr(visual_chain.settings, "thermal_gate_enabled", False)
        monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (40.0, 1.0))
        called: list[bool] = []
        monkeypatch.setattr(
            visual_chain,
            "evaluate_thermal_gate",
            lambda: called.append(True) or visual_chain.ThermalVerdict(
                state="hot", temp_c=40.0, age_sec=1.0, allows_gpu_work=False, reason="x"
            ),
        )
        _no_gpu_allowed(monkeypatch)
        persisted = _persist_spy(monkeypatch)

        # With the flag off the gate must not even be consulted, and the run
        # must reach the GPU request. `_no_gpu_allowed` raises there, and the
        # chain converts that into a generation_failed readout -- so reaching
        # "generation_failed" IS the proof the gate was bypassed.
        chain = asyncio.run(visual_chain.run_visual_chain_once(bus=None))

        assert called == [], "gate was consulted despite being disabled"
        assert chain.terminal_reason == "generation_failed"
        assert not any(getattr(c, "terminal_reason", None) == "thermal_refused" for c in persisted)


class TestSensorReader:
    def test_reads_temp_and_age_from_the_live_payload_shape(self, monkeypatch) -> None:
        """Shape copied verbatim from the running hub on 2026-08-30."""
        payload = {
            "ok": True,
            "age_sec": 0.673327,
            "snapshot": {
                "status": "ok",
                "frame": {
                    "environment": {
                        "temp_c": 30.74,
                        "humidity_pct": 23.68,
                        "pressure_hpa": 871.55,
                    }
                },
            },
        }
        monkeypatch.setattr(visual_chain.json, "loads", lambda _raw: payload)

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"{}"

        monkeypatch.setattr(visual_chain.urllib.request, "urlopen", lambda *a, **k: _Resp())
        assert visual_chain.read_cabinet_temp_c() == (30.74, 0.673327)

    def test_a_failing_endpoint_returns_no_reading_rather_than_raising(self, monkeypatch) -> None:
        def _boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(visual_chain.urllib.request, "urlopen", _boom)
        assert visual_chain.read_cabinet_temp_c() == (None, None)
