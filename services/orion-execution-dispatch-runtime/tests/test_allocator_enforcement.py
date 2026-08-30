"""Cover for the allocator-enforcement path in `_send_prepared_candidates`.

This code had ZERO tests when it shipped, and that is the direct cause of an
`UnboundLocalError` reaching a live container: `allocation` was assigned only
inside `if motor is not None:` and read unconditionally below. The three
existing test files in this directory never touch the send loop.

These test the decision logic at the smallest honest boundary rather than
standing up the whole worker: the failure modes are about which candidates
survive filtering, and about a name being bound.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1] / "app" / "worker.py"


def _send_function() -> ast.FunctionDef:
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "_send_prepared_candidates":
                return node  # type: ignore[return-value]
    raise AssertionError("_send_prepared_candidates not found")


class TestAllocationIsAlwaysBound:
    """The exact defect that shipped. `allocation` must be bound on every path
    that reaches its read, including the one where no motor budget exists."""

    def test_allocation_is_assigned_before_the_motor_budget_branch(self) -> None:
        func = _send_function()
        lines = ast.unparse(func).splitlines()

        unconditional = next(
            (i for i, line in enumerate(lines) if line.strip() == "allocation = None"),
            None,
        )
        assert unconditional is not None, (
            "allocation must be bound unconditionally; without it a tick with no "
            "motor budget raises UnboundLocalError and takes save_dispatch_frame "
            "down with it"
        )

        branch = next(
            (i for i, line in enumerate(lines) if "motor = self._derive_motor_budget" in line),
            None,
        )
        assert branch is not None
        assert unconditional < branch, "the binding must precede the branch that may skip it"

    def test_the_enforcement_read_is_guarded_against_none(self) -> None:
        """Belt as well as braces: even bound, None must not be read as
        'the allocator refused everything'."""
        source = ast.unparse(_send_function())
        assert "allocation is not None" in source


class TestProbeExemption:
    """A tripwire probe is not Orion choosing to act; it is the system testing
    whether it still can. If the allocator can refuse it, the tick abandons,
    probe successes never accrue, and the tripwire can never re-arm -- the
    self-sealing defect of the 2026-08-23 outage, by a new route."""

    def test_a_probe_in_flight_clears_the_admitted_filter(self) -> None:
        source = ast.unparse(_send_function())
        assert "self._tripwire_probe_in_flight" in source, "probe exemption is missing entirely"

        marker = "if allocator_admitted_ids is not None and self._tripwire_probe_in_flight"
        assert marker in source.replace("\n", " ") or "_tripwire_probe_in_flight:" in source

    def test_the_exemption_comes_after_the_filter_is_built(self) -> None:
        """Exempting before the set is built would be a no-op the next
        assignment overwrites.

        Anchored on the line that CLEARS the filter, not on any mention of the
        probe flag -- the flag is also read much earlier when the probe slot is
        claimed, and matching that occurrence made this assertion compare the
        wrong two lines.
        """
        lines = ast.unparse(_send_function()).splitlines()
        built = next(i for i, l in enumerate(lines) if "allocator_admitted_ids = {" in l)
        exempt = next(
            i
            for i, l in enumerate(lines)
            if "allocator_admitted_ids = None" in l and i > built
        )
        assert built < exempt


class TestColdStartCost:
    """Without a cold-start cost the enforcing allocator is an absorbing state:
    a never-run action is refused `no_cost_estimate` before the information
    floor is consulted, and the only way to earn a cost is to run."""

    def test_a_missing_measured_cost_falls_back_to_the_typical_cost(self) -> None:
        source = WORKER.read_text(encoding="utf-8")
        assert "orion_dispatch_motor_typical_cost_sec" in source, (
            "no cold-start cost: a new action can never be admitted, so the "
            "documented escape hatch ('better actions, declared signals') is "
            "structurally unreachable under enforcement"
        )

    def test_the_fallback_is_on_the_allocator_candidate_not_elsewhere(self) -> None:
        source = WORKER.read_text(encoding="utf-8")
        start = source.index("candidate_from_dispatch(")
        end = source.index("allocation = allocate(", start)
        assert "orion_dispatch_motor_typical_cost_sec" in source[start:end]


class TestRefusedEverythingSignal:
    def test_the_counter_resets_rather_than_accumulating_forever(self) -> None:
        """A lifetime total cannot distinguish 'refusing everything right now'
        from 'refused everything once last week'."""
        source = WORKER.read_text(encoding="utf-8")
        assert "self._all_refused_consecutive_ticks = 0" in source
        assert "self._all_refused_consecutive_ticks += 1" in source

    def test_it_is_logged_loudly_enough_to_find(self) -> None:
        source = WORKER.read_text(encoding="utf-8")
        index = source.index("motor_allocator_refused_everything")
        preceding = source[max(0, index - 400) : index]
        assert "logger.warning" in preceding, (
            "an all-refused dispatcher must not be indistinguishable from an "
            "idle one at INFO level"
        )


@pytest.mark.parametrize(
    "name",
    ["motor_allocator_enforced", "motor_allocator_refused_everything", "motor_allocator_exempt"],
)
def test_every_enforcement_branch_announces_itself(name: str) -> None:
    """Each distinct outcome of the enforcement path needs its own greppable
    line; otherwise 'why did Orion send nothing' has no answer in the logs."""
    assert name in WORKER.read_text(encoding="utf-8")
