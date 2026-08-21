"""Gate: config/vision_profiles.yaml's documented vram_budget must match the
env keys scheduler.py actually enforces.

Why this exists. On 2026-08-20 orion-vision-host stopped serving every task
with `gpu_hard_floor` and stayed down ~21 hours; Orion was blind and nothing
alerted. app/gpu.py computes

    effective_free = g.free_mb - reserve_mb        # then compared to hard_floor

and the budget in force was reserve=3500 / hard_floor=1400.

**The config never drifted. The hardware moved.** Those values are the
originals from the first vision-host commit (626c103ee), written when athena
carried a **P100 16GB**, and they were correct there: 16384 - ~3238MB resident
= ~13.1GB free, minus 3500 reserve = ~9.6GB, comfortably above the 1400 floor.
The P100 later moved to circe and athena was left with a **Tesla P4, 7.68GB**.
Nobody re-derived the budget, and on the smaller card the same numbers are
arithmetically unsatisfiable the moment the models warm: 4191 - 3500 = 691 <
1400, forever. The service stayed `Up` and healthy and served nothing.

That is why this file does NOT hardcode a card. A constant baked in against
today's GPU is the exact failure being guarded: it would go stale the next time
a card moves, silently, in the same way. `test_budget_is_satisfiable_on_this_
host` reads the REAL GPU and asserts the budget works on the smallest card
actually present.

The yaml block is separately read by nothing (app/main.py:64 takes the value
from settings/env), so it is pure prose that operators copy. The equality test
makes that prose load-bearing instead of decorative.

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


def _smallest_gpu_total_mb() -> int | None:
    """Total VRAM of the smallest GPU actually present, or None if unreadable."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except Exception:
        return None
    totals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
    return min(totals) if totals else None


def test_budget_is_satisfiable_on_this_host() -> None:
    """The floors must be satisfiable on the card this host ACTUALLY has.

    Reads real hardware rather than a baked-in constant, because a baked-in
    constant is precisely what failed: the budget was written for a P100 16GB,
    the card was swapped for a P4 7.68GB, and nothing re-derived it. A test
    pinned to "the P4" would rot the same way on the next swap.

    RESIDENT_MB is the measured warm footprint (GroundingDINO fp32 + SigLIP2 +
    the caption VLM), 3238MB on 2026-08-21. It is a floor on what the service
    holds once warm, not a guess -- but it is the one number here that still
    has to be re-measured when the model set changes, which is why it is named
    and dated rather than inlined.
    """
    import pytest

    total_mb = _smallest_gpu_total_mb()
    if total_mb is None:
        pytest.skip("no readable GPU on this host; nothing to check the budget against")

    budget = yaml.safe_load(_PROFILES.read_text())["runtime"]["vram_budget"]

    RESIDENT_MB = 3238  # measured 2026-08-21, athena/P4
    free_when_warm = total_mb - RESIDENT_MB
    effective_free = free_when_warm - int(budget["reserve_mb"])

    assert effective_free > int(budget["hard_floor_mb"]), (
        f"Smallest GPU on this host is {total_mb}MB. With the warm model set "
        f"resident (~{RESIDENT_MB}MB) that leaves ~{free_when_warm}MB free; "
        f"reserve_mb={budget['reserve_mb']} leaves {effective_free}MB effective, "
        f"at or below hard_floor_mb={budget['hard_floor_mb']}. app/gpu.py would "
        f"refuse every task with gpu_hard_floor while the container reports "
        f"healthy -- the 2026-08-20 outage, exactly. Re-derive the budget for "
        f"this card (see this module's docstring)."
    )
