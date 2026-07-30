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
