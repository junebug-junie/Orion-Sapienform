"""Deterministic block tariff: marginal $/kWh and block-split energy cost.

Blocks reset each billing cycle, so the price of the next kWh depends on how much
the house has already used this cycle. That position is the whole reason run cost
is priced here and not as kWh x an average rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


class TariffError(ValueError):
    """Tariff config is structurally invalid."""


@dataclass(frozen=True)
class Block:
    up_to_kwh: Optional[float]
    usd_per_kwh: float


@dataclass(frozen=True)
class Season:
    name: str
    months: frozenset[int]
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Tariff:
    version: str
    cost_basis: str
    seasons: tuple[Season, ...]
    energy_multiplier: float
    fixed_monthly_usd: float

    def season_for(self, month: int) -> Season:
        for season in self.seasons:
            if month in season.months:
                return season
        raise TariffError(f"no season covers month {month}")

    def energy_cost_usd(self, kwh: float, *, cycle_kwh_before: float, month: int) -> float:
        remaining = _require_finite_non_negative("kwh", kwh)
        position = _require_finite_non_negative("cycle_kwh_before", cycle_kwh_before)
        if remaining == 0.0:
            return 0.0
        base = 0.0
        for block in self.season_for(month).blocks:
            if remaining <= 0.0:
                break
            if block.up_to_kwh is not None and position >= block.up_to_kwh:
                continue
            room = remaining if block.up_to_kwh is None else min(remaining, block.up_to_kwh - position)
            base += room * block.usd_per_kwh
            remaining -= room
            position += room
        return base * self.energy_multiplier

    def marginal_usd_per_kwh(self, *, cycle_kwh: float, month: int) -> float:
        position = _require_finite_non_negative("cycle_kwh", cycle_kwh)
        for block in self.season_for(month).blocks:
            if block.up_to_kwh is None or position < block.up_to_kwh:
                return block.usd_per_kwh * self.energy_multiplier
        raise TariffError("tariff has no open top block")


def _require_finite_non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise TariffError(f"{name} must be finite, got {value!r}")
    if number < 0.0:
        raise TariffError(f"{name} must be non-negative, got {value!r}")
    return number


def _require_mapping(raw: Any, *, context: str) -> dict:
    if not isinstance(raw, dict):
        raise TariffError(f"{context} must be a mapping, got {type(raw).__name__}")
    return raw


def _require_key(raw: dict, key: str, *, context: str) -> Any:
    if key not in raw:
        raise TariffError(f"{context} missing required key {key!r}")
    return raw[key]


def _require_number(raw: dict, key: str, *, context: str) -> float:
    value = _require_key(raw, key, context=context)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TariffError(f"{context} key {key!r} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise TariffError(f"{context} key {key!r} must be finite, got {value!r}")
    return number


def _season(name: str, raw: dict) -> Season:
    blocks: list[Block] = []
    for index, item in enumerate(raw.get("blocks") or []):
        block_raw = _require_mapping(item, context=f"season {name!r} block {index}")
        up_to_raw = block_raw.get("up_to_kwh")
        up_to = None if up_to_raw is None else _require_number(
            block_raw, "up_to_kwh", context=f"season {name!r} block {index}"
        )
        rate = _require_number(block_raw, "cents_per_kwh", context=f"season {name!r} block {index}") / 100.0
        blocks.append(Block(up_to, rate))
    if not blocks or blocks[-1].up_to_kwh is not None:
        raise TariffError(f"season {name!r} must end with an open (up_to_kwh: null) block")
    bounds = [b.up_to_kwh for b in blocks[:-1]]
    if any(b is None for b in bounds):
        raise TariffError(f"season {name!r} blocks must ascend with only the last open")
    if any(b <= 0.0 for b in bounds):
        raise TariffError(f"season {name!r} block bounds must be > 0")
    if bounds != sorted(set(bounds)):
        raise TariffError(f"season {name!r} blocks must strictly ascend with only the last open")
    return Season(name=name, months=frozenset(int(m) for m in raw.get("months") or []), blocks=tuple(blocks))


def load_tariff(path: str | Path) -> Tariff:
    loaded = yaml.safe_load(Path(path).read_text())
    raw = _require_mapping(loaded, context="tariff document")
    version = str(_require_key(raw, "tariff_version", context="tariff document"))
    if "cost_basis" in raw:
        cost_basis = str(raw["cost_basis"])
        if cost_basis != "pre_tax":
            raise TariffError(f"unsupported cost_basis {cost_basis!r}; only 'pre_tax' is supported")
    else:
        cost_basis = "pre_tax"
    seasons = tuple(_season(name, body) for name, body in (raw.get("seasons") or {}).items())
    covered = sorted(m for s in seasons for m in s.months)
    if covered != list(range(1, 13)):
        raise TariffError(f"seasons must cover each month 1-12 exactly once, got {covered}")
    adj = _require_mapping(raw.get("energy_adjustments") or {}, context="energy_adjustments")
    stage1 = sum(
        _require_number(
            _require_mapping(item, context=f"energy_adjustments.on_base_pct[{index}]"),
            "pct",
            context=f"energy_adjustments.on_base_pct[{index}]",
        )
        for index, item in enumerate(adj.get("on_base_pct") or [])
    )
    stage2 = sum(
        _require_number(
            _require_mapping(item, context=f"energy_adjustments.on_adjusted_pct[{index}]"),
            "pct",
            context=f"energy_adjustments.on_adjusted_pct[{index}]",
        )
        for index, item in enumerate(adj.get("on_adjusted_pct") or [])
    )
    fixed_monthly_usd = sum(
        _require_number(
            _require_mapping(item, context=f"fixed_monthly[{index}]"),
            "usd",
            context=f"fixed_monthly[{index}]",
        )
        for index, item in enumerate(raw.get("fixed_monthly") or [])
    )
    return Tariff(
        version=version,
        cost_basis=cost_basis,
        seasons=seasons,
        energy_multiplier=(1.0 + stage1 / 100.0) * (1.0 + stage2 / 100.0),
        fixed_monthly_usd=fixed_monthly_usd,
    )
