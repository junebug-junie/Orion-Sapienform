"""The single-flight lock must be impossible to hold indefinitely.

The lock makes overlapping GPU runs impossible, which is correct. What it did
NOT do was bound how long one run could hold it: every hop has its own timeout,
but per-hop timeouts sum rather than bound, and a `urlopen` socket timeout is
reset by each chunk received, so a peer that dribbles bytes is never caught by
one. Observed live 2026-08-31: `already_in_flight` returned while circe was
completely idle, cleared only by a container restart.

That matters more than it looks. `express` -- Orion's only outward action -- is
dispatched by the motor allocator through this lock. A held lock silently
un-schedules it: no exception, no failed dispatch, just a `no-op` that reads
exactly like healthy busy-ness forever.

The property under test is therefore not "the deadline fires" but "a hung run
does not cost the NEXT run", plus "a busy signal can state its own age".
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app import visual_chain


@pytest.fixture(autouse=True)
def _gate_off_and_lock_free(monkeypatch):
    # The thermal gate is a separate concern with its own suite, and it is
    # checked BEFORE the lock -- leaving it live here would let a hot office
    # make these tests pass without the lock ever being taken.
    monkeypatch.setattr(visual_chain.settings, "thermal_gate_enabled", False)
    monkeypatch.setattr(visual_chain.settings, "visual_chain_run_deadline_sec", 0.25)
    assert not visual_chain._visual_chain_lock.locked(), "lock leaked from an earlier test"
    yield
    assert not visual_chain._visual_chain_lock.locked(), "this test leaked the lock"


def _persist_spy(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        visual_chain, "persist_reverie_visual_chain", lambda c: persisted.append(c) or True
    )
    return persisted


def _hanging_body(monkeypatch, *, hang_sec: float = 30.0):
    """A body that never returns within the deadline. 30s, not `Event().wait()`,
    so a broken deadline fails this suite by timing out loudly rather than
    hanging pytest forever."""

    async def _hang(*args, **kwargs):
        await asyncio.sleep(hang_sec)
        raise AssertionError("the hung body was allowed to finish")

    monkeypatch.setattr(visual_chain, "_run_visual_chain_body", _hang)


class TestDeadlineReleasesTheLock:
    def test_a_hung_run_does_not_wedge_the_next_one(self, monkeypatch) -> None:
        """THE regression test. Before the deadline existed, call two returned
        None forever."""
        _persist_spy(monkeypatch)
        _hanging_body(monkeypatch)

        async def scenario():
            first = await visual_chain.run_visual_chain_once(bus=None)
            # The second call must actually RUN, not no-op. Swapping in a body
            # that returns a sentinel is what distinguishes "the lock was
            # released" from "the second call also timed out", which would
            # produce the same terminal_reason and hide the difference.
            ran = asyncio.Event()

            async def _quick(*args, **kwargs):
                ran.set()
                return visual_chain.ReverieVisualChainV1(
                    chain_id="second", created_at=visual_chain._now()
                )

            monkeypatch.setattr(visual_chain, "_run_visual_chain_body", _quick)
            second = await visual_chain.run_visual_chain_once(bus=None)
            return first, second, ran.is_set()

        first, second, second_ran = asyncio.run(scenario())

        assert first is not None
        assert first.terminal_reason == "run_deadline_exceeded"
        assert second_ran, "the lock was still held: the second run never started"
        assert second is not None and second.chain_id == "second"

    def test_the_abandoned_run_is_recorded_not_just_logged(self, monkeypatch) -> None:
        # Same argument as the thermal refusal: a run that leaves no row is
        # indistinguishable from a worker that died.
        persisted = _persist_spy(monkeypatch)
        _hanging_body(monkeypatch)

        chain = asyncio.run(visual_chain.run_visual_chain_once(bus=None))

        assert persisted == [chain]
        assert chain.chain_json["run_deadline_sec"] == 0.25
        assert chain.chain_json["held_sec"] >= 0.25, "held_sec must be the real elapsed time"
        # Upper bound too. `>=` alone is guaranteed by wait_for's construction, so
        # a regression firing the deadline at 29s instead of 0.25s would still
        # pass -- the suite would only get slow, which is exactly the symptom
        # nobody notices (review finding).
        assert chain.chain_json["held_sec"] < 1.0, "the deadline fired far later than configured"

    def test_a_body_that_raises_also_releases_the_lock(self, monkeypatch) -> None:
        """The deadline is not the only exit. A body raising must not leave the
        lock held either -- and `async with` is what guarantees that, so this
        fails if anyone converts it to a manual acquire()."""
        _persist_spy(monkeypatch)

        async def _boom(*args, **kwargs):
            raise RuntimeError("hop exploded")

        monkeypatch.setattr(visual_chain, "_run_visual_chain_body", _boom)

        with pytest.raises(RuntimeError):
            asyncio.run(visual_chain.run_visual_chain_once(bus=None))

        assert not visual_chain._visual_chain_lock.locked()
        # The MODULE GLOBAL, not the helper. Review finding: the helper
        # short-circuits on `not lock.locked()`, so asserting through it passes
        # whether or not the `finally` reset exists -- three assertions in this
        # file were vacuous for exactly that reason.
        assert visual_chain._visual_chain_started_at is None
        assert visual_chain.visual_chain_in_flight_for() is None


class TestBusySignalCanStateItsAge:
    def test_in_flight_for_is_none_when_free(self) -> None:
        assert visual_chain.visual_chain_in_flight_for() is None

    def test_in_flight_for_reports_elapsed_while_held(self, monkeypatch) -> None:
        _persist_spy(monkeypatch)
        seen: list[float | None] = []

        async def _observe_from_inside(*args, **kwargs):
            await asyncio.sleep(0.05)
            seen.append(visual_chain.visual_chain_in_flight_for())
            return visual_chain.ReverieVisualChainV1(
                chain_id="obs", created_at=visual_chain._now()
            )

        monkeypatch.setattr(visual_chain, "_run_visual_chain_body", _observe_from_inside)
        asyncio.run(visual_chain.run_visual_chain_once(bus=None))

        assert seen and seen[0] is not None
        assert seen[0] >= 0.05, "the age must be real elapsed time, not a constant"
        # And it must go back to None, or the NEXT call's log line reports a
        # stale age from a run that already finished. Asserted on the global for
        # the reason given above -- through the helper this is vacuous.
        assert visual_chain._visual_chain_started_at is None

    def test_a_concurrent_caller_no_ops_and_the_age_is_readable(self, monkeypatch) -> None:
        _persist_spy(monkeypatch)
        monkeypatch.setattr(visual_chain.settings, "visual_chain_run_deadline_sec", 5.0)
        held_seen: list[float | None] = []
        release = asyncio.Event()

        async def scenario():
            async def _slow(*args, **kwargs):
                await release.wait()
                return visual_chain.ReverieVisualChainV1(
                    chain_id="slow", created_at=visual_chain._now()
                )

            monkeypatch.setattr(visual_chain, "_run_visual_chain_body", _slow)
            first = asyncio.create_task(visual_chain.run_visual_chain_once(bus=None))
            await asyncio.sleep(0.05)
            second = await visual_chain.run_visual_chain_once(bus=None)
            held_seen.append(visual_chain.visual_chain_in_flight_for())
            release.set()
            return second, await first

        second, first = asyncio.run(scenario())

        assert second is None, "a concurrent caller must no-op, not queue a second GPU run"
        assert first is not None and first.chain_id == "slow"
        assert held_seen[0] is not None and held_seen[0] >= 0.05


class TestTheDeadlineIsNotRedundant:
    def test_deadline_is_longer_than_the_sum_of_hop_timeouts(self) -> None:
        """A deadline below the hop budget would abandon slow-but-working runs
        and destroy real images. Recomputed from the live settings rather than
        hardcoded, so raising a hop timeout past the deadline fails here."""
        from app.settings import ThoughtSettings

        s = ThoughtSettings()
        hop_budget = (
            s.visual_chain_interpretation_timeout_sec
            + s.visual_chain_diffusion_timeout_sec
            + s.visual_chain_percept_upload_timeout_sec
            + s.visual_chain_caption_timeout_sec
        )
        assert s.visual_chain_run_deadline_sec > hop_budget, (
            f"deadline {s.visual_chain_run_deadline_sec}s must exceed the {hop_budget}s "
            "hop budget or a working run gets abandoned mid-generation"
        )


class TestTheHandlerCannotWedgeTheLockItself:
    """The sharpest review finding: the abandoned-run persist sits INSIDE the
    lock and OUTSIDE the wait_for, so before it was bounded, the code written to
    release a wedged lock could wedge it in exactly the same way -- and
    `suppress(Exception)` is no defence, because the failure mode is hanging,
    not raising.
    """

    def test_a_hanging_persist_still_releases_the_lock(self, monkeypatch) -> None:
        monkeypatch.setattr(visual_chain, "_DEADLINE_PERSIST_TIMEOUT_SEC", 0.25)

        # A real blocking call in a real worker thread -- the way a Postgres call
        # with no statement_timeout blocks -- not an awaitable that can be
        # cancelled. Released explicitly at the end so the abandoned thread does
        # not stall interpreter shutdown for 30s on every run of this suite.
        import threading

        stuck = threading.Event()

        def _persist_that_hangs(chain):
            stuck.wait(30)
            return True

        monkeypatch.setattr(visual_chain, "persist_reverie_visual_chain", _persist_that_hangs)
        _hanging_body(monkeypatch)

        async def scenario():
            t0 = time.monotonic()
            first = await visual_chain.run_visual_chain_once(bus=None)
            elapsed = time.monotonic() - t0
            assert not visual_chain._visual_chain_lock.locked(), (
                "the deadline handler's own persist held the lock -- the fix "
                "reproduced the bug it was written to fix"
            )
            # TIME-BOUNDED, and this is the whole assertion. Without it the test
            # is vacuous: a hanging persist eventually returns when the stuck
            # event times out, so every assertion below still passes -- the suite
            # just silently takes 30s instead of 4s. Mutation-tested: reverting
            # the handler's persist to an unbounded `to_thread` passed all 11
            # tests and only got slower, which is precisely the failure mode
            # this file exists to catch.
            assert elapsed < 3.0, (
                f"the abandoned-run persist held the lock for {elapsed:.1f}s -- it is "
                "not bounded by _DEADLINE_PERSIST_TIMEOUT_SEC"
            )

            async def _quick(*args, **kwargs):
                return visual_chain.ReverieVisualChainV1(
                    chain_id="after-hanging-persist", created_at=visual_chain._now()
                )

            monkeypatch.setattr(visual_chain, "_run_visual_chain_body", _quick)
            second = await visual_chain.run_visual_chain_once(bus=None)
            # Released HERE, not after asyncio.run returns: loop shutdown joins
            # the default executor, so an abandoned thread still blocked on this
            # event would add its full 30s wait to every run of this suite.
            stuck.set()
            return first, second

        try:
            first, second = asyncio.run(scenario())
        finally:
            stuck.set()

        assert first.terminal_reason == "run_deadline_exceeded"
        assert second is not None and second.chain_id == "after-hanging-persist"

    def test_an_abandoned_run_points_at_the_row_the_body_may_have_written(
        self, monkeypatch
    ) -> None:
        """Cancellation can land after the body committed its own chain row, so
        one run leaves two rows with different ids. Without this key nothing
        links them and the hub renders them as unrelated events."""
        _persist_spy(monkeypatch)

        async def _hang_with_an_id(*args, **kwargs):
            visual_chain._visual_chain_body_chain_id = "body-side-id"
            await asyncio.sleep(30)

        monkeypatch.setattr(visual_chain, "_run_visual_chain_body", _hang_with_an_id)

        chain = asyncio.run(visual_chain.run_visual_chain_once(bus=None))

        assert chain.chain_json["abandoned_chain_id"] == "body-side-id"
        assert chain.chain_id != "body-side-id", "the two rows are distinct rows"
        # And it must not leak into the NEXT abandonment.
        assert visual_chain._visual_chain_body_chain_id is None


class TestTheWedgeIsAudible:
    def test_a_run_past_its_own_deadline_logs_at_warning(self, monkeypatch, caplog) -> None:
        """The one wedge path the deadline cannot close is a non-cancellable
        hop, and this WARNING is its only signal. An INFO line here would sit at
        the same level as the ordinary busy case, which is what let the original
        wedge go unnoticed."""
        monkeypatch.setattr(visual_chain.settings, "visual_chain_run_deadline_sec", 0.01)
        release = asyncio.Event()

        async def scenario():
            async def _slow(*args, **kwargs):
                await release.wait()
                return visual_chain.ReverieVisualChainV1(
                    chain_id="slow", created_at=visual_chain._now()
                )

            monkeypatch.setattr(visual_chain, "_run_visual_chain_body", _slow)
            # Hold the lock directly so the run outlives the deadline WITHOUT the
            # wait_for firing -- i.e. simulate the residual path, not the fixed one.
            async with visual_chain._visual_chain_lock:
                visual_chain._visual_chain_started_at = 0.0  # monotonic epoch: very old
                with caplog.at_level("WARNING"):
                    result = await visual_chain.run_visual_chain_once(bus=None)
            visual_chain._visual_chain_started_at = None
            release.set()
            return result

        result = asyncio.run(scenario())

        assert result is None, "a concurrent caller must still no-op"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "a run past its own deadline must not be logged at INFO"
        assert "PAST its" in warnings[0].getMessage()


class TestTheEndpointReportsTheAge:
    def test_already_in_flight_response_carries_held_sec(self, monkeypatch) -> None:
        """The response contract, not just the log line. `express` bounces off
        this endpoint, so the age has to survive the HTTP hop to be worth
        anything -- and the consumer allowlist in orion-cortex-exec dropped it
        until a review caught that."""
        import json
        import sys

        from app import main as thought_main

        # Resolve the module the ENDPOINT will actually use. `app.main` imports
        # `.visual_chain` lazily inside the handler, and this service's conftest
        # purges `app.*` from sys.modules, so the module object this test file
        # imported at collection time is NOT necessarily the one the handler
        # resolves. Getting that wrong here does not fail safe: an earlier
        # version of this test held a lock on the wrong module object and the
        # handler ran a real generation against the live diffusion host.
        vc = sys.modules["app.visual_chain"]

        def _no_body(*args, **kwargs):
            raise AssertionError("the body must never run: the lock is held")

        monkeypatch.setattr(vc, "_run_visual_chain_body", _no_body)

        class _FakeBus:
            def __init__(self, url=None):
                pass

            async def connect(self):
                return None

            async def close(self):
                return None

        monkeypatch.setattr(
            "orion.core.bus.async_service.OrionBusAsync", _FakeBus, raising=False
        )

        async def scenario():
            # Hold the lock directly rather than racing a second task: the branch
            # under test is reached purely by the lock being held.
            async with vc._visual_chain_lock:
                monkeypatch.setattr(vc, "_visual_chain_started_at", time.monotonic())
                resp = await thought_main.visual_chain_run_once()
            monkeypatch.setattr(vc, "_visual_chain_started_at", None)
            return json.loads(resp.body)

        payload = asyncio.run(scenario())

        assert payload["ran"] is False
        assert payload["reason"] == "already_in_flight"
        assert "held_sec" in payload, "the endpoint must report the age, not just log it"
        assert payload["held_sec"] is not None and payload["held_sec"] >= 0.0
