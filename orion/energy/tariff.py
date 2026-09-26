"""Deterministic block tariff: marginal $/kWh and block-split energy cost.

Blocks reset each billing cycle, so the price of the next kWh depends on how much
the house has already used this cycle. That position is the whole reason run cost
is priced here and not as kWh x an average rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
        remaining = max(0.0, float(kwh))
        position = max(0.0, float(cycle_kwh_before))
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
        position = max(0.0, float(cycle_kwh))
        for block in self.season_for(month).blocks:
            if block.up_to_kwh is None or position < block.up_to_kwh:
                return block.usd_per_kwh * self.energy_multiplier
        raise TariffError("tariff has no open top block")


def _season(name: str, raw: dict) -> Season:
    blocks: list[Block] = []
    for item in raw.get("blocks") or []:
        up_to = item.get("up_to_kwh")
        blocks.append(Block(None if up_to is None else float(up_to), float(item["cents_per_kwh"]) / 100.0))
    if not blocks or blocks[-1].up_to_kwh is not None:
        raise TariffError(f"season {name!r} must end with an open (up_to_kwh: null) block")
    bounds = [b.up_to_kwh for b in blocks[:-1]]
    if any(b is None for b in bounds) or bounds != sorted(bounds):
        raise TariffError(f"season {name!r} blocks must ascend with only the last open")
    return Season(name=name, months=frozenset(int(m) for m in raw.get("months") or []), blocks=tuple(blocks))


def load_tariff(path: str | Path) -> Tariff:
    raw = yaml.safe_load(Path(path).read_text())
    seasons = tuple(_season(name, body) for name, body in (raw.get("seasons") or {}).items())
    covered = sorted(m for s in seasons for m in s.months)
    if covered != list(range(1, 13)):
        raise TariffError(f"seasons must cover each month 1-12 exactly once, got {covered}")
    adj = raw.get("energy_adjustments") or {}
    stage1 = sum(float(a["pct"]) for a in adj.get("on_base_pct") or [])
    stage2 = sum(float(a["pct"]) for a in adj.get("on_adjusted_pct") or [])
    return Tariff(
        version=str(raw["tariff_version"]),
        cost_basis=str(raw.get("cost_basis", "pre_tax")),
        seasons=seasons,
        energy_multiplier=(1.0 + stage1 / 100.0) * (1.0 + stage2 / 100.0),
        fixed_monthly_usd=sum(float(f["usd"]) for f in raw.get("fixed_monthly") or []),
    )
