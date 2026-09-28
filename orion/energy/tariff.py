"""Deterministic block tariff: marginal $/kWh and block-split energy cost.

Blocks reset each billing cycle, so the price of the next kWh depends on how much
the house has already used this cycle. That position is the whole reason run cost
is priced here and not as kWh x an average rate.

A tariff with `proration` scales its block sizes and monthly charges by
period_days / base_days, as RMP does on the bill (400 kWh and $12.00 per 30 days
printed as 493 kWh and $14.80 for a 37-day period). Without it, callers' period
lengths are ignored and every period prices like a flat month.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Optional

import yaml

BASE_ENERGY = "base_energy"
CUSTOMER_CHARGE = "customer_charge"


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
class Rider:
    """A percent charge on the sum of named components (base_energy, customer_charge, earlier riders)."""

    name: str
    pct: float
    applies_to: tuple[str, ...]


@dataclass(frozen=True)
class FixedCharge:
    name: str
    usd: float
    component: Optional[str] = None
    tax_exempt: bool = False


@dataclass(frozen=True)
class BillEstimate:
    """A period priced the way the bill prints it: every line rounded to the cent."""

    lines: tuple[tuple[str, float], ...]
    energy_charge: float
    customer_charge: float
    adjustments: float
    fees: float
    credits: float
    pre_tax_usd: float
    taxes: Optional[float]
    total_usd: float


RESERVED_LINE_NAMES = frozenset({BASE_ENERGY, CUSTOMER_CHARGE, "sales_tax"})


def _dec(value: float, places: str = "1e-9") -> Decimal:
    """Exact decimal of a config number; quantized so a float like 0.09833199999999999 reads 0.098332."""
    return Decimal(repr(float(value))).quantize(Decimal(places), rounding=ROUND_HALF_UP)


def _cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class Tariff:
    version: str
    cost_basis: str
    seasons: tuple[Season, ...]
    energy_multiplier: float
    fixed_monthly_usd: float
    # Sums of fixed_items / per_bill_items / riders, precomputed by load_tariff.
    customer_charge_monthly_usd: float = 0.0
    customer_multiplier: float = 1.0
    per_bill_usd: float = 0.0
    proration_base_days: Optional[float] = None
    round_block_kwh: bool = False
    sales_tax_pct: Optional[float] = None
    riders: tuple[Rider, ...] = ()
    fixed_items: tuple[FixedCharge, ...] = ()
    per_bill_items: tuple[FixedCharge, ...] = ()

    def season_for(self, month: int) -> Season:
        for season in self.seasons:
            if month in season.months:
                return season
        raise TariffError(f"no season covers month {month}")

    def _scale(self, period_days: Optional[float]) -> float:
        if self.proration_base_days is None or period_days is None:
            return 1.0
        return _require_positive("period_days", period_days) / self.proration_base_days

    def _bounds(self, season: Season, period_days: Optional[float]) -> list[Optional[float]]:
        scale = self._scale(period_days)
        out: list[Optional[float]] = []
        for block in season.blocks:
            if block.up_to_kwh is None:
                out.append(None)
                continue
            bound = block.up_to_kwh * scale
            if self.round_block_kwh:
                bound = float(_dec(bound).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            out.append(bound)
        return out

    def _split(
        self, kwh: float, *, cycle_kwh_before: float, month: int, period_days: Optional[float]
    ) -> list[tuple[float, float]]:
        """(kWh, $/kWh) slices of `kwh` starting at cycle position `cycle_kwh_before`, pre-riders."""
        remaining = _require_finite_non_negative("kwh", kwh)
        position = _require_finite_non_negative("cycle_kwh_before", cycle_kwh_before)
        season = self.season_for(month)
        slices: list[tuple[float, float]] = []
        for block, up_to in zip(season.blocks, self._bounds(season, period_days)):
            if remaining <= 0.0:
                break
            if up_to is not None and position >= up_to:
                continue
            room = remaining if up_to is None else min(remaining, up_to - position)
            slices.append((room, block.usd_per_kwh))
            remaining -= room
            position += room
        return slices

    def energy_cost_usd(
        self, kwh: float, *, cycle_kwh_before: float, month: int, period_days: Optional[float] = None
    ) -> float:
        slices = self._split(kwh, cycle_kwh_before=cycle_kwh_before, month=month, period_days=period_days)
        return sum(k * rate for k, rate in slices) * self.energy_multiplier

    def marginal_usd_per_kwh(self, *, cycle_kwh: float, month: int, period_days: Optional[float] = None) -> float:
        position = _require_finite_non_negative("cycle_kwh", cycle_kwh)
        season = self.season_for(month)
        for block, up_to in zip(season.blocks, self._bounds(season, period_days)):
            if up_to is None or position < up_to:
                return block.usd_per_kwh * self.energy_multiplier
        raise TariffError("tariff has no open top block")

    @property
    def has_customer_charge(self) -> bool:
        return any(item.component == CUSTOMER_CHARGE for item in self.fixed_items)

    def customer_charge_usd(self, period_days: Optional[float] = None) -> float:
        """The customer charge as printed (prorated, before riders)."""
        return self.customer_charge_monthly_usd * self._scale(period_days)

    def fixed_usd(self, period_days: Optional[float] = None) -> float:
        """Everything not priced per kWh, pre-tax: prorated monthly charges, their riders, per-bill items."""
        scale = self._scale(period_days)
        customer_riders = self.customer_charge_monthly_usd * scale * (self.customer_multiplier - 1.0)
        return self.fixed_monthly_usd * scale + customer_riders + self.per_bill_usd

    def itemize(self, kwh: float, *, month: int, period_days: float) -> BillEstimate:
        """Price a whole period's kWh line by line, rounding each line to the cent like the bill.

        Decimal throughout, so an exact half cent (7.63% of $50.00 = 3.815) rounds up as printed
        rather than down off a float like 3.81499999. One season for the whole period (a bill
        spanning a season change is not modeled).
        """
        lines: list[tuple[str, Decimal]] = []
        blocks = self._split(kwh, cycle_kwh_before=0.0, month=month, period_days=period_days)
        energy = Decimal(0)
        for index, (k, rate) in enumerate(blocks):
            usd = _cents(_dec(k, "1e-6") * _dec(rate))
            lines.append((f"energy_block_{index + 1}", usd))
            energy += usd
        if self.proration_base_days is None:
            scale = Decimal(1)
        else:
            scale = _dec(_require_positive("period_days", period_days)) / _dec(self.proration_base_days)
        components: dict[str, Decimal] = {BASE_ENERGY: energy, CUSTOMER_CHARGE: Decimal(0)}
        fees = Decimal(0)
        exempt = Decimal(0)
        for item in self.fixed_items:
            usd = _cents(_dec(item.usd) * scale)
            lines.append((item.name, usd))
            if item.component == CUSTOMER_CHARGE:
                components[CUSTOMER_CHARGE] += usd
            else:
                fees += usd
            if item.tax_exempt:
                exempt += usd
        adjustments = Decimal(0)
        for rider in self.riders:
            usd = _cents(_dec(rider.pct) / 100 * sum((components[c] for c in rider.applies_to), Decimal(0)))
            lines.append((rider.name, usd))
            components[rider.name] = usd
            adjustments += usd
        credits = Decimal(0)
        for item in self.per_bill_items:
            usd = _cents(_dec(item.usd))
            lines.append((item.name, usd))
            credits += usd
            if item.tax_exempt:
                exempt += usd
        pre_tax = energy + components[CUSTOMER_CHARGE] + fees + adjustments + credits
        taxes = None if self.sales_tax_pct is None else _cents(_dec(self.sales_tax_pct) / 100 * (pre_tax - exempt))
        if taxes is not None:
            lines.append(("sales_tax", taxes))
        return BillEstimate(
            lines=tuple((name, float(usd)) for name, usd in lines),
            energy_charge=float(energy),
            customer_charge=float(components[CUSTOMER_CHARGE]),
            adjustments=float(adjustments),
            fees=float(fees),
            credits=float(credits),
            pre_tax_usd=float(pre_tax),
            taxes=None if taxes is None else float(taxes),
            total_usd=float(pre_tax + (taxes or Decimal(0))),
        )


def _require_finite_non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise TariffError(f"{name} must be finite, got {value!r}")
    if number < 0.0:
        raise TariffError(f"{name} must be non-negative, got {value!r}")
    return number


def _require_positive(name: str, value: float) -> float:
    number = _require_finite_non_negative(name, value)
    if number == 0.0:
        raise TariffError(f"{name} must be > 0, got {value!r}")
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


def _require_bool(raw: dict, key: str, *, context: str, default: bool = False) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise TariffError(f"{context} key {key!r} must be true or false, got {value!r}")
    return value


def _require_name(raw: dict, *, context: str, default: Optional[str] = None) -> str:
    value = raw.get("name", default)
    if not isinstance(value, str) or not value:
        raise TariffError(f"{context} name must be a non-empty string, got {value!r}")
    return value


def _pct_items(raw: Any, key: str) -> list[dict]:
    return [_require_mapping(item, context=f"{key}[{i}]") for i, item in enumerate(raw or [])]


def _legacy_riders(adj: dict) -> list[Rider]:
    """`energy_adjustments`: stage 1 on base energy, stage 2 on base + stage 1 (compounding)."""
    stage1 = [
        Rider(str(item.get("name", f"on_base_{i}")), _require_number(item, "pct", context=f"on_base_pct[{i}]"),
              (BASE_ENERGY,))
        for i, item in enumerate(_pct_items(adj.get("on_base_pct"), "energy_adjustments.on_base_pct"))
    ]
    adjusted = (BASE_ENERGY, *(r.name for r in stage1))
    stage2 = [
        Rider(str(item.get("name", f"on_adjusted_{i}")),
              _require_number(item, "pct", context=f"on_adjusted_pct[{i}]"), adjusted)
        for i, item in enumerate(_pct_items(adj.get("on_adjusted_pct"), "energy_adjustments.on_adjusted_pct"))
    ]
    return stage1 + stage2


def _riders(raw: Any) -> list[Rider]:
    out: list[Rider] = []
    for i, item in enumerate(_pct_items(raw, "riders")):
        context = f"riders[{i}]"
        name = _require_name(item, context=context)
        applies = item.get("applies_to")
        if not isinstance(applies, list) or not applies:
            raise TariffError(f"{context} applies_to must be a non-empty list")
        if len(set(applies)) != len(applies):
            raise TariffError(f"{context} applies_to repeats a component (it would be counted twice)")
        out.append(Rider(name, _require_number(item, "pct", context=context), tuple(str(a) for a in applies)))
    return out


def _multipliers(riders: list[Rider]) -> tuple[float, float]:
    """Each rider is linear in (base energy $, customer charge $); sum the coefficients."""
    coef: dict[str, tuple[float, float]] = {BASE_ENERGY: (1.0, 0.0), CUSTOMER_CHARGE: (0.0, 1.0)}
    energy, customer = 1.0, 1.0
    for rider in riders:
        if rider.name in coef:
            raise TariffError(f"rider name {rider.name!r} is duplicated or reserved")
        unknown = [c for c in rider.applies_to if c not in coef]
        if unknown:
            raise TariffError(f"rider {rider.name!r} applies_to unknown or later component(s) {unknown}")
        e = rider.pct / 100.0 * sum(coef[c][0] for c in rider.applies_to)
        c = rider.pct / 100.0 * sum(coef[c][1] for c in rider.applies_to)
        coef[rider.name] = (e, c)
        energy += e
        customer += c
    if energy <= 0.0 or customer <= 0.0:
        raise TariffError(f"riders net to a non-positive multiplier (energy {energy}, customer {customer})")
    return energy, customer


def _charges(raw: Any, key: str) -> list[FixedCharge]:
    out: list[FixedCharge] = []
    for i, item in enumerate(_pct_items(raw, key)):
        context = f"{key}[{i}]"
        component = item.get("component")
        allowed = (None, CUSTOMER_CHARGE) if key == "fixed_monthly" else (None,)
        if component not in allowed:
            raise TariffError(f"{context} component must be one of {allowed}, got {component!r}")
        out.append(FixedCharge(
            name=_require_name(item, context=context, default=f"{key}_{i}"),
            usd=_require_number(item, "usd", context=context),
            component=component,
            tax_exempt=_require_bool(item, "tax_exempt", context=context),
        ))
    return out


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
    if "riders" in raw and "energy_adjustments" in raw:
        raise TariffError("use either riders or energy_adjustments, not both")
    if "riders" in raw:
        riders = _riders(raw["riders"])
    else:
        riders = _legacy_riders(_require_mapping(raw.get("energy_adjustments") or {}, context="energy_adjustments"))
    energy_multiplier, customer_multiplier = _multipliers(riders)
    fixed_items = _charges(raw.get("fixed_monthly"), "fixed_monthly")
    per_bill_items = _charges(raw.get("per_bill"), "per_bill")
    has_customer_charge = any(item.component == CUSTOMER_CHARGE for item in fixed_items)
    for rider in riders:
        if CUSTOMER_CHARGE in rider.applies_to and not has_customer_charge:
            raise TariffError(f"rider {rider.name!r} applies to customer_charge but no fixed_monthly item is one")
    names = [r.name for r in riders] + [c.name for c in fixed_items] + [c.name for c in per_bill_items]
    clashes = sorted(
        {n for n in names if names.count(n) > 1}
        | {n for n in names if n in RESERVED_LINE_NAMES or n.startswith("energy_block_")}
    )
    if clashes:
        raise TariffError(f"charge names must be unique and not reserved: {clashes}")
    proration_base_days: Optional[float] = None
    round_block_kwh = False
    if raw.get("proration") is not None:
        pro = _require_mapping(raw["proration"], context="proration")
        proration_base_days = _require_number(pro, "base_days", context="proration")
        if proration_base_days <= 0.0:
            raise TariffError("proration base_days must be > 0")
        round_block_kwh = _require_bool(pro, "round_block_kwh", context="proration")
    sales_tax_pct: Optional[float] = None
    if raw.get("sales_tax") is not None:
        sales_tax_pct = _require_number(_require_mapping(raw["sales_tax"], context="sales_tax"), "pct",
                                        context="sales_tax")
    return Tariff(
        version=version,
        cost_basis=cost_basis,
        seasons=seasons,
        energy_multiplier=energy_multiplier,
        fixed_monthly_usd=sum(item.usd for item in fixed_items),
        customer_charge_monthly_usd=sum(item.usd for item in fixed_items if item.component == CUSTOMER_CHARGE),
        customer_multiplier=customer_multiplier,
        per_bill_usd=sum(item.usd for item in per_bill_items),
        proration_base_days=proration_base_days,
        round_block_kwh=round_block_kwh,
        sales_tax_pct=sales_tax_pct,
        riders=tuple(riders),
        fixed_items=tuple(fixed_items),
        per_bill_items=tuple(per_bill_items),
    )
