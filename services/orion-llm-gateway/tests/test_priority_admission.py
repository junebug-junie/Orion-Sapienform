from __future__ import annotations

import asyncio
from typing import Any, List, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app import priority_admission
from app.llm_backend import RouteTarget


@pytest.fixture(autouse=True)
def _reset_semaphores() -> None:
    priority_admission._semaphores.clear()
    yield
    priority_admission._semaphores.clear()


def _target(**overrides: Any) -> RouteTarget:
    defaults = dict(url="http://quick:8013", served_by="atlas-worker-fast-1", backend="llamacpp")
    defaults.update(overrides)
    return RouteTarget(**defaults)


class TestFreeSlotCount:
    @pytest.mark.asyncio
    async def test_counts_idle_slots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slots = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": True},
            {"id": 2, "is_processing": False},
            {"id": 3, "is_processing": False},
        ]
        mock_response = httpx.Response(
            200, json=slots, request=httpx.Request("GET", "http://quick:8013/slots")
        )

        async def _fake_get(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            assert url == "http://quick:8013/slots"
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
        free = await priority_admission._free_slot_count("http://quick:8013")
        assert free == 3

    @pytest.mark.asyncio
    async def test_returns_none_when_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_get(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("refused", request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
        free = await priority_admission._free_slot_count("http://quick:8013")
        assert free is None

    @pytest.mark.asyncio
    async def test_returns_none_for_non_list_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_response = httpx.Response(
            200, json={"error": "slots disabled"}, request=httpx.Request("GET", "http://quick:8013/slots")
        )

        async def _fake_get(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
        free = await priority_admission._free_slot_count("http://quick:8013")
        assert free is None


class TestWaitForSlack:
    @pytest.mark.asyncio
    async def test_returns_true_immediately_when_slack_already_available(self) -> None:
        with patch.object(priority_admission, "_free_slot_count", AsyncMock(return_value=3)):
            ok = await priority_admission.wait_for_slack(
                _target(reserved_free_slots=2), poll_interval_sec=0.01, max_wait_sec=1.0
            )
        assert ok is True

    @pytest.mark.asyncio
    async def test_polls_until_slack_frees_up(self) -> None:
        calls = {"n": 0}

        async def _sequence(base_url: str, **kwargs: Any) -> Optional[int]:
            calls["n"] += 1
            return 0 if calls["n"] < 3 else 2

        with patch.object(priority_admission, "_free_slot_count", _sequence):
            ok = await priority_admission.wait_for_slack(
                _target(reserved_free_slots=2), poll_interval_sec=0.01, max_wait_sec=1.0
            )
        assert ok is True
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_times_out_and_returns_false_when_permanently_busy(self) -> None:
        with patch.object(priority_admission, "_free_slot_count", AsyncMock(return_value=0)):
            ok = await priority_admission.wait_for_slack(
                _target(reserved_free_slots=2), poll_interval_sec=0.01, max_wait_sec=0.05
            )
        assert ok is False

    @pytest.mark.asyncio
    async def test_returns_false_immediately_when_slots_unreachable(self) -> None:
        with patch.object(priority_admission, "_free_slot_count", AsyncMock(return_value=None)):
            ok = await priority_admission.wait_for_slack(
                _target(reserved_free_slots=2), poll_interval_sec=0.01, max_wait_sec=5.0
            )
        assert ok is False

    @pytest.mark.asyncio
    async def test_defaults_reserved_free_slots_when_unset(self) -> None:
        with patch.object(priority_admission, "_free_slot_count", AsyncMock(return_value=1)) as mocked:
            ok = await priority_admission.wait_for_slack(
                _target(reserved_free_slots=None), poll_interval_sec=0.01, max_wait_sec=1.0
            )
        assert ok is True
        mocked.assert_awaited()


class TestBackgroundAdmissionConcurrency:
    @pytest.mark.asyncio
    async def test_caps_concurrent_admissions_per_route(self) -> None:
        active = 0
        max_active = 0
        order: List[str] = []

        async def _always_free(base_url: str, **kwargs: Any) -> int:
            return 5

        async def _hold(tag: str) -> None:
            nonlocal active, max_active
            async with priority_admission.background_admission(
                "aitown", _target(reserved_free_slots=1),
                concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0,
            ):
                active += 1
                max_active = max(max_active, active)
                order.append(f"enter:{tag}")
                await asyncio.sleep(0.05)
                active -= 1
                order.append(f"exit:{tag}")

        with patch.object(priority_admission, "_free_slot_count", _always_free):
            await asyncio.gather(_hold("a"), _hold("b"))

        assert max_active == 1
        assert order == ["enter:a", "exit:a", "enter:b", "exit:b"]

    @pytest.mark.asyncio
    async def test_different_route_keys_get_independent_semaphores(self) -> None:
        active = 0
        max_active = 0

        async def _always_free(base_url: str, **kwargs: Any) -> int:
            return 5

        async def _hold(route_key: str) -> None:
            nonlocal active, max_active
            async with priority_admission.background_admission(
                route_key, _target(reserved_free_slots=1),
                concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0,
            ):
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.05)
                active -= 1

        with patch.object(priority_admission, "_free_slot_count", _always_free):
            await asyncio.gather(_hold("aitown"), _hold("some_other_lane"))

        assert max_active == 2


class TestBackgroundAdmissionReleasesOnFailure:
    """Regression coverage for a real bug caught in review: if __aenter__
    raises after acquiring the semaphore (an exception during wait_for_slack,
    or the enclosing task being cancelled mid-wait), __aexit__ never runs --
    so without an explicit release-on-failure, the permit leaks forever and
    wedges that route's background lane at its concurrency cap."""

    @pytest.mark.asyncio
    async def test_semaphore_released_when_wrapped_body_raises(self) -> None:
        async def _always_free(base_url: str, **kwargs: Any) -> int:
            return 5

        target = _target(reserved_free_slots=1)
        with patch.object(priority_admission, "_free_slot_count", _always_free):
            with pytest.raises(RuntimeError):
                async with priority_admission.background_admission(
                    "aitown", target, concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0,
                ):
                    raise RuntimeError("boom")

            # A second acquire on the same route must not block -- if the
            # first permit leaked, this would hang until asyncio.wait_for's
            # timeout below fires.
            async def _try_acquire() -> None:
                async with priority_admission.background_admission(
                    "aitown", target, concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0,
                ):
                    pass

            await asyncio.wait_for(_try_acquire(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_semaphore_released_when_cancelled_mid_wait(self) -> None:
        target = _target(reserved_free_slots=1)

        async def _never_free(base_url: str, **kwargs: Any) -> int:
            await asyncio.sleep(10)  # long enough to guarantee cancellation lands first
            return 0

        async def _enter_and_wait() -> None:
            async with priority_admission.background_admission(
                "aitown", target, concurrency=1, poll_interval_sec=0.01, max_wait_sec=5.0,
            ):
                pass  # never reached -- cancelled while still waiting for slack

        with patch.object(priority_admission, "_free_slot_count", _never_free):
            task = asyncio.create_task(_enter_and_wait())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        async def _always_free(base_url: str, **kwargs: Any) -> int:
            return 5

        async def _try_acquire() -> None:
            async with priority_admission.background_admission(
                "aitown", target, concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0,
            ):
                pass

        with patch.object(priority_admission, "_free_slot_count", _always_free):
            # If the earlier cancellation leaked the permit, this hangs and
            # asyncio.wait_for raises TimeoutError instead of completing.
            await asyncio.wait_for(_try_acquire(), timeout=1.0)


class TestAdmissionIsRecordedInTheLedger:
    """ROADMAP A5: every admission decision must reach the ledger, not just the log.

    Six call sites emit an admission outcome. These drive the real functions rather than
    asserting on `_log_and_record` directly, because the failure this guards against is a call
    site that logs and forgets to record -- which a test of the helper in isolation cannot see.
    """

    @pytest.fixture(autouse=True)
    def _fresh_ledger(self, monkeypatch: pytest.MonkeyPatch):
        from app.admission_ledger import AdmissionLedger

        led = AdmissionLedger()
        monkeypatch.setattr(priority_admission, "get_ledger", lambda: led)
        yield led

    @pytest.mark.asyncio
    async def test_immediate_admit_records_a_non_deferral(self, _fresh_ledger, monkeypatch):
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=4))
        assert await priority_admission.wait_for_slack(
            _target(reserved_free_slots=2), poll_interval_sec=0.01, max_wait_sec=1.0,
            route_key="quick_background",
        ) is True
        snap = _fresh_ledger.snapshot(window_s=600.0)
        assert (snap["checked"], snap["deferrals"]) == (1, 0)
        assert snap["routes"] == ["quick_background"]

    @pytest.mark.asyncio
    async def test_a_real_wait_records_a_deferral_with_its_duration(self, _fresh_ledger, monkeypatch):
        """Busy, busy, then room: two slept intervals, so polls=3 and it IS a deferral."""
        monkeypatch.setattr(
            priority_admission, "_free_slot_count", AsyncMock(side_effect=[0, 1, 4])
        )
        assert await priority_admission.wait_for_slack(
            _target(reserved_free_slots=2), poll_interval_sec=0.01, max_wait_sec=5.0,
            route_key="quick_background",
        ) is True
        snap = _fresh_ledger.snapshot(window_s=600.0)
        assert (snap["checked"], snap["deferrals"]) == (1, 1)
        assert snap["longest_wait_s"] > 0.0
        assert snap["last_deferral_ts"] is not None

    @pytest.mark.asyncio
    async def test_unreachable_slots_records_unchecked(self, _fresh_ledger, monkeypatch):
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=None))
        assert await priority_admission.wait_for_slack(
            _target(), poll_interval_sec=0.01, max_wait_sec=1.0,
        ) is False
        snap = _fresh_ledger.snapshot(window_s=600.0)
        assert (snap["checked"], snap["unchecked"], snap["deferrals"]) == (1, 1, 0)

    @pytest.mark.asyncio
    async def test_timeout_records_a_deferral(self, _fresh_ledger, monkeypatch):
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=0))
        assert await priority_admission.wait_for_slack(
            _target(reserved_free_slots=2), poll_interval_sec=0.01, max_wait_sec=0.0,
        ) is False
        snap = _fresh_ledger.snapshot(window_s=600.0)
        assert (snap["deferrals"], snap["timeouts"]) == (1, 1)

    @pytest.mark.asyncio
    async def test_bus_path_records_as_via_bus(self, _fresh_ledger, monkeypatch):
        """The bus path (run_llm_chat's caller in main.py) carries 100% of live background
        traffic (2026-08-19). Since 2026-09-05 it uses this same context manager with
        via="bus" -- the deleted blocking variant used to be the only thing recording it."""
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=4))
        async with priority_admission.background_admission(
            "quick_background", _target(reserved_free_slots=2),
            concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0, via="bus",
        ):
            pass
        assert _fresh_ledger.snapshot(window_s=600.0, via="bus")["checked"] == 1
        assert _fresh_ledger.snapshot(window_s=600.0, via="http")["checked"] == 0

    @pytest.mark.asyncio
    async def test_background_admission_passes_its_route_key_through(self, _fresh_ledger, monkeypatch):
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=4))
        async with priority_admission.background_admission(
            "quick_background", _target(reserved_free_slots=2),
            concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0,
        ):
            pass
        assert _fresh_ledger.snapshot(window_s=600.0)["routes"] == ["quick_background"]

    @pytest.mark.asyncio
    async def test_a_broken_ledger_never_breaks_a_dispatch(self, monkeypatch):
        """Bookkeeping is not allowed to cost a request. Fail-open is the whole contract here."""
        class _Exploding:
            def record(self, **kw):
                raise RuntimeError("ledger on fire")

        monkeypatch.setattr(priority_admission, "get_ledger", lambda: _Exploding())
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=4))
        assert await priority_admission.wait_for_slack(
            _target(reserved_free_slots=2), poll_interval_sec=0.01, max_wait_sec=1.0,
        ) is True


class TestSemaphoreQueueIsRecorded:
    """REVIEW FIX (A5). `background_admission` acquires its permit BEFORE polling /slots, and
    LLM_GATEWAY_BACKGROUND_CONCURRENCY defaults to 1. A second concurrent background request
    therefore blocks on the semaphore for the whole of the first one's generation, then finds
    room on its first poll -- recorded, until this fix, as `polls=1 admitted`: a request that
    waited for the length of a generation, reported as one that never waited.
    """

    @pytest.fixture(autouse=True)
    def _fresh_ledger(self, monkeypatch: pytest.MonkeyPatch):
        from app.admission_ledger import AdmissionLedger

        led = AdmissionLedger()
        monkeypatch.setattr(priority_admission, "get_ledger", lambda: led)
        yield led

    @pytest.mark.asyncio
    async def test_the_second_concurrent_request_records_a_deferral(self, _fresh_ledger, monkeypatch):
        """Two callers, concurrency=1, /slots always reports room. The first is not a deferral;
        the second is, purely because it queued -- no /slots poll would ever reveal it."""
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=4))
        target = _target(reserved_free_slots=2)
        release = asyncio.Event()

        async def _holder():
            async with priority_admission.background_admission(
                "quick_background", target, concurrency=1,
                poll_interval_sec=0.01, max_wait_sec=1.0,
            ):
                await release.wait()

        held = asyncio.create_task(_holder())
        await asyncio.sleep(0.05)          # let the holder take the only permit

        async def _waiter():
            async with priority_admission.background_admission(
                "quick_background", target, concurrency=1,
                poll_interval_sec=0.01, max_wait_sec=1.0,
            ):
                pass

        queued = asyncio.create_task(_waiter())
        await asyncio.sleep(0.15)          # the waiter is now blocked in acquire()
        release.set()
        await asyncio.gather(held, queued)

        snap = _fresh_ledger.snapshot(window_s=600.0)
        assert snap["checked"] == 2
        assert snap["queued"] == 1
        assert snap["deferrals"] == 1, "the queued request must be recorded as a deferral"
        assert snap["longest_wait_s"] >= 0.1, "and its duration must be the queue time"

    @pytest.mark.asyncio
    async def test_an_uncontended_acquire_is_not_a_deferral(self, _fresh_ledger, monkeypatch):
        """The complement: `locked()` is exact, so an idle semaphore must not mark anything."""
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=4))
        async with priority_admission.background_admission(
            "quick_background", _target(reserved_free_slots=2),
            concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0,
        ):
            pass
        snap = _fresh_ledger.snapshot(window_s=600.0)
        assert (snap["queued"], snap["deferrals"]) == (0, 0)

    @pytest.mark.asyncio
    async def test_the_two_call_paths_are_distinguishable(self, _fresh_ledger, monkeypatch):
        """`quick_background` carries AI Town's NPC dialogue (http) and Orion's own work (bus).
        The cue makes a first-person claim, so the ledger must keep them apart."""
        monkeypatch.setattr(priority_admission, "_free_slot_count", AsyncMock(return_value=4))
        for via in ("http", "bus"):
            async with priority_admission.background_admission(
                "quick_background", _target(reserved_free_slots=2),
                concurrency=1, poll_interval_sec=0.01, max_wait_sec=1.0, via=via,
            ):
                pass
        assert _fresh_ledger.snapshot(window_s=600.0, via="http")["checked"] == 1
        assert _fresh_ledger.snapshot(window_s=600.0, via="bus")["checked"] == 1
