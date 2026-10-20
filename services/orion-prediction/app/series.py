"""Series registry: config/prediction/series.v1.yaml -> validated SeriesSpec objects.

All durations are stored in whole minutes. Validation is strict: a bad registry refuses to load
rather than forecasting something nobody declared.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_DUR = re.compile(r"^\s*(\d+)\s*(min|h|d)\s*$")
_ID = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_KEY = re.compile(r"^[a-z0-9_]{1,40}$")
_UNIT_MIN = {"min": 1, "h": 60, "d": 1440}
TRANSFORMS = ("none", "log")
AGGS = ("mean", "max", "min", "last")


class SeriesConfigError(ValueError):
    pass


def duration_min(value: Any) -> int:
    m = _DUR.match(str(value))
    if not m:
        raise SeriesConfigError(f"bad duration {value!r} (want '<n>min', '<n>h' or '<n>d')")
    out = int(m.group(1)) * _UNIT_MIN[m.group(2)]
    if out <= 0:
        raise SeriesConfigError(f"duration must be positive: {value!r}")
    return out


@dataclass(frozen=True)
class Seasonality:
    name: str
    period_min: int
    fourier_order: int


@dataclass(frozen=True)
class Threshold:
    name: str
    value: float
    window_min: int
    max_probability: float | None = None


@dataclass(frozen=True)
class Source:
    kind: str  # biometrics | cooling
    node: str | None = None
    key: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    role: str
    source: Source
    unit: str
    grain_min: int
    agg: str
    max_fill_min: int
    consumers: tuple[str, ...]
    transform: str = "none"
    train_from: datetime | None = None
    train_window_min: int = 21 * 1440
    seasonalities: tuple[Seasonality, ...] = ()
    exclude: tuple[tuple[datetime, datetime], ...] = ()
    horizons_min: tuple[int, ...] = ()
    thresholds: tuple[Threshold, ...] = ()
    covariates: tuple[str, ...] = ()
    # Pending change-event breaks (app/events.py): Prophet fits a level shift at each one.
    breaks: tuple[datetime, ...] = ()

    @property
    def max_horizon_steps(self) -> int:
        return max(self.horizons_min) // self.grain_min

    def steps(self, minutes: int) -> int:
        return minutes // self.grain_min

    def threshold(self, name: str) -> Threshold | None:
        return next((t for t in self.thresholds if t.name == name), None)


def _ts(value: Any, where: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise SeriesConfigError(f"{where}: bad timestamp {value!r}") from exc
    if dt.tzinfo is None:
        raise SeriesConfigError(f"{where}: timestamp needs a timezone: {value!r}")
    return dt.astimezone(timezone.utc)


def _multiple(minutes: int, grain: int, where: str) -> int:
    if minutes % grain:
        raise SeriesConfigError(f"{where}: {minutes} min is not a multiple of the {grain} min grain")
    return minutes


def _source(raw: dict, sid: str) -> Source:
    kind = raw.get("kind")
    if kind == "biometrics":
        node, key = raw.get("node"), raw.get("key")
        if not node or not key or not _KEY.match(str(key)):
            raise SeriesConfigError(f"{sid}: biometrics source needs node and a [a-z0-9_] key")
        return Source(kind="biometrics", node=str(node), key=str(key))
    if kind == "cooling":
        if not raw.get("role"):
            raise SeriesConfigError(f"{sid}: cooling source needs role")
        return Source(kind="cooling", role=str(raw["role"]))
    raise SeriesConfigError(f"{sid}: unknown source kind {kind!r}")


def _spec(raw: dict) -> SeriesSpec:
    sid = str(raw.get("series_id", ""))
    if not _ID.match(sid):
        raise SeriesConfigError(f"bad series_id {sid!r}")
    role = raw.get("role")
    if role not in ("target", "covariate"):
        raise SeriesConfigError(f"{sid}: role must be target or covariate")
    consumers = tuple(str(c) for c in raw.get("consumers") or ())
    if not consumers:
        raise SeriesConfigError(f"{sid}: consumers is empty -- an unread series does not belong here")
    grain = duration_min(raw.get("grain", "5min"))
    agg = raw.get("agg", "mean")
    if agg not in AGGS:
        raise SeriesConfigError(f"{sid}: agg must be one of {AGGS}")
    transform = raw.get("transform", "none")
    if transform not in TRANSFORMS:
        raise SeriesConfigError(f"{sid}: transform must be one of {TRANSFORMS}")
    seas = []
    for s in raw.get("seasonalities") or ():
        period = _multiple(duration_min(s["period"]), grain, f"{sid}.seasonalities.{s.get('name')}")
        order = int(s.get("fourier_order", 3))
        if order < 1:
            raise SeriesConfigError(f"{sid}: fourier_order must be >= 1")
        seas.append(Seasonality(name=str(s["name"]), period_min=period, fourier_order=order))
    if len({s.name for s in seas}) != len(seas):
        raise SeriesConfigError(f"{sid}: duplicate seasonality names")
    horizons = tuple(sorted(_multiple(duration_min(h), grain, f"{sid}.horizons") for h in raw.get("horizons") or ()))
    if role == "target" and not horizons:
        raise SeriesConfigError(f"{sid}: a target needs horizons")
    if len(set(horizons)) != len(horizons):
        raise SeriesConfigError(f"{sid}: duplicate horizons")
    thresholds = []
    for t in raw.get("thresholds") or ():
        mp = t.get("max_probability")
        thresholds.append(Threshold(
            name=str(t["name"]), value=float(t["value"]),
            window_min=_multiple(duration_min(t["window"]), grain, f"{sid}.thresholds.{t['name']}"),
            max_probability=None if mp is None else float(mp)))
    exclude = []
    for e in raw.get("exclude") or ():
        start, end = _ts(e["start"], f"{sid}.exclude"), _ts(e["end"], f"{sid}.exclude")
        if end <= start:
            raise SeriesConfigError(f"{sid}: exclude window ends before it starts")
        exclude.append((start, end))
    return SeriesSpec(
        series_id=sid, role=role, source=_source(raw.get("source") or {}, sid), unit=str(raw.get("unit", "")),
        grain_min=grain, agg=agg, max_fill_min=duration_min(raw.get("max_fill", "15min")), consumers=consumers,
        transform=transform,
        train_from=_ts(raw["train_from"], f"{sid}.train_from") if raw.get("train_from") else None,
        train_window_min=duration_min(raw.get("train_window", "21d")),
        seasonalities=tuple(seas), exclude=tuple(exclude), horizons_min=horizons,
        thresholds=tuple(thresholds), covariates=tuple(str(c) for c in raw.get("covariates") or ()),
    )


@dataclass
class Registry:
    series: dict[str, SeriesSpec] = field(default_factory=dict)

    def targets(self) -> list[SeriesSpec]:
        return [s for s in self.series.values() if s.role == "target"]

    def get(self, series_id: str) -> SeriesSpec | None:
        return self.series.get(series_id)


def parse_registry(doc: dict) -> Registry:
    if (doc or {}).get("version") != 1:
        raise SeriesConfigError("registry version must be 1")
    reg = Registry()
    for raw in doc.get("series") or ():
        spec = _spec(raw)
        if spec.series_id in reg.series:
            raise SeriesConfigError(f"duplicate series_id {spec.series_id}")
        reg.series[spec.series_id] = spec
    for spec in reg.series.values():
        for cov in spec.covariates:
            other = reg.series.get(cov)
            if other is None:
                raise SeriesConfigError(f"{spec.series_id}: unknown covariate {cov}")
            if other.grain_min != spec.grain_min:
                raise SeriesConfigError(f"{spec.series_id}: covariate {cov} has a different grain")
    return reg


def load_registry(path: str | Path) -> Registry:
    return parse_registry(yaml.safe_load(Path(path).read_text()))
