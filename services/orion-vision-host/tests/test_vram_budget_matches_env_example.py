"""Gate: config/vision_profiles.yaml's documented vram_budget must match the
env keys scheduler.py actually enforces.

Why this exists. On 2026-08-20 orion-vision-host stopped serving every task
with `gpu_hard_floor` and stayed down ~21 hours; Orion was blind and nothing
alerted. Root cause was not a code bug -- it was that the *documentation-only*
`runtime.vram_budget` block in config/vision_profiles.yaml carried
3500/2200/1400, someone copied those into a live
services/orion-vision-host/.env, and app/gpu.py's

    effective_free = g.free_mb - reserve_mb        # then compared to hard_floor

is unsatisfiable at those numbers on the 7.68GB Tesla P4 once the ~3.2GB of
warm resident models load: 4191 - 3500 = 691 < 1400, forever.

The yaml block is read by nothing (app/main.py:64 takes the value from
settings/env), so it is pure prose -- which is exactly why it drifted and
exactly why it was trusted. This test makes the prose load-bearing instead of
decorative: the doc and the enforced contract cannot diverge again silently.

It deliberately does NOT assert against the local .env (gitignored, and a
host-specific override is legitimate) -- only against .env_example, which is
the checked-in operator contract (AGENTS.md section 7).
"""

from __future__ import annotations

import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PROFILES = _REPO / "config" / "vision_profiles.yaml"
_ENV_EXAMPLE = _REPO / "services" / "orion-vision-host" / ".env_example"

_PAIRS = {
    "reserve_mb": "VISION_VRAM_RESERVE_MB",
    "soft_floor_mb": "VISION_VRAM_SOFT_FLOOR_MB",
    "hard_floor_mb": "VISION_VRAM_HARD_FLOOR_MB",
}


def _env_example_ints() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in _ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key in _PAIRS.values():
            out[key] = int(value.strip())
    return out


def test_vram_budget_block_matches_env_example() -> None:
    budget = yaml.safe_load(_PROFILES.read_text())["runtime"]["vram_budget"]
    env = _env_example_ints()

    missing = [k for k in _PAIRS.values() if k not in env]
    assert not missing, f".env_example is missing VRAM keys: {missing}"

    for yaml_key, env_key in _PAIRS.items():
        assert int(budget[yaml_key]) == env[env_key], (
            f"config/vision_profiles.yaml runtime.vram_budget.{yaml_key}="
            f"{budget[yaml_key]} but {env_key}={env[env_key]} in .env_example. "
            "These must match -- the env key is enforced, the yaml is what "
            "operators read and copy. See this module's docstring for the "
            "21-hour outage this drift caused."
        )


def test_reserve_leaves_room_for_resident_models_on_the_smallest_card() -> None:
    """The floors must be satisfiable on the card this service actually runs on.

    Live-measured, not assumed: the Tesla P4 is 7680MB total and the warm
    resident set (GroundingDINO + SigLIP2 + the VLM) measured 3238MB on
    2026-08-21. A config that cannot pick the GPU with the models already
    loaded is a config that bricks the service the moment it warms up, which
    is precisely the failure this file exists to prevent -- and it is not
    caught by the equality test above, since .env_example could drift to bad
    values in lockstep with the yaml.
    """
    budget = yaml.safe_load(_PROFILES.read_text())["runtime"]["vram_budget"]

    P4_TOTAL_MB = 7680
    MEASURED_RESIDENT_MB = 3238
    free_when_warm = P4_TOTAL_MB - MEASURED_RESIDENT_MB

    effective_free = free_when_warm - int(budget["reserve_mb"])
    assert effective_free > int(budget["hard_floor_mb"]), (
        f"With models warm the P4 has ~{free_when_warm}MB free; "
        f"reserve_mb={budget['reserve_mb']} leaves {effective_free}MB effective, "
        f"which is at or below hard_floor_mb={budget['hard_floor_mb']}. "
        "app/gpu.py would refuse every task with gpu_hard_floor and the service "
        "would serve nothing while looking healthy."
    )
