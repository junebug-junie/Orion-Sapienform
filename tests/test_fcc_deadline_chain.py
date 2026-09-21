"""The FCC motor budget is not a standalone number.

Raising HARNESS_FCC_TIMEOUT_SEC without the Hub / curiosity / durable-runs
waiters above it makes the motor believe it has more time while a caller
kills the turn earlier and throws the work away. Confirmed twice:

- 2026-08-26: Hub RPC at 960s abandoned a finished finalize five seconds
  after the verdict published.
- 2026-09-19: a 2h motor budget with the old 3600s Hub hard ceiling would
  still have died at one hour.

This reads the operator-contract files (not Field defaults) so a comment-only
edit cannot hide a broken chain.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _example_float(rel: str, key: str) -> float:
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith(f"{key}="):
            return float(line.split("=", 1)[1])
    raise AssertionError(f"{rel} has no {key}=")


def test_callers_sit_above_the_fcc_motor_budget() -> None:
    fcc = _example_float(
        "services/orion-harness-governor/.env_example", "HARNESS_FCC_TIMEOUT_SEC"
    )
    rpc = _example_float(
        "services/orion-hub/.env_example", "HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC"
    )
    max_wait = _example_float(
        "services/orion-hub/.env_example", "HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC"
    )
    curiosity = _example_float(
        "services/orion-hub/.env_example", "HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC"
    )
    durable = _example_float(
        "services/orion-durable-runs/.env_example", "DURABLE_RUNS_TURN_RPC_TIMEOUT_SEC"
    )

    finalize_chain_sec = (
        _example_float(
            "services/orion-harness-governor/.env_example", "SUBSTRATE_FINALIZE_TIMEOUT_SEC"
        )
        + _example_float(
            "services/orion-harness-governor/.env_example", "FINALIZE_REFLECT_TIMEOUT_SEC"
        )
        + _example_float(
            "services/orion-harness-governor/.env_example", "RESPONSE_REPAIR_TIMEOUT_SEC"
        )
    )
    stance_sec = 400.0

    assert rpc >= fcc + finalize_chain_sec, (
        f"Hub RPC {rpc} must cover FCC {fcc} + finalize {finalize_chain_sec}"
    )
    assert max_wait >= rpc, (
        f"Hub hard ceiling {max_wait} must be >= soft RPC {rpc} "
        "(the client clamps the first wait to max_wait)"
    )
    assert curiosity >= rpc + stance_sec, (
        f"Curiosity outer wait {curiosity} must cover stance {stance_sec} + RPC {rpc}"
    )
    assert durable >= curiosity, (
        f"Durable-runs RPC {durable} must sit above curiosity {curiosity}"
    )
