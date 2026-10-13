"""Runtime immune sweep: has one of Orion's own gauges stopped telling the truth? Pure, I/O-free.

Temporal Self rev 4 (PR #2369), section R4. The producer is the regulation step, gated on idle
(``services/orion-durable-runs/app/temporal_self_graph.py``, ``immune`` node); this module only
judges series it is handed and never fixes anything.

Scope (v1): field channels whose glossary entry carries a ``semantics:`` block
(``config/field/field_channel_glossary.v1.yaml``, read through ``orion.field.channel_glossary``),
read from ``substrate_field_state`` at one sample per minute. It extends that semantic layer; it
is not a registry of its own.

Four failures, each with the reason it is a lie and not a calm reading:

* ``decayed_to_zero`` -- successive changes are exact integer powers of one ratio ``r`` in (0, 1)
  (relative tolerance ``decay_rel_tol`` on the log ratio), for at least ``decay_min_changes``
  changes. A real measurement does not move by an exact power of a constant; a staleness-decay
  loop does (``node:substrate.route`` 2026-07-26: 0.92 per tick for 48 h). Minute sampling sees
  ``r ** k`` with a varying tick count ``k``, so the base is searched as ``smallest step / d`` for
  ``d`` up to ``decay_max_divisor``. Applies to every value kind but ``placeholder`` (the route
  failure was a ``count``: a count cannot legitimately take the values ``0.92 ** n`` at all).
* ``replayed`` -- the same value outside rest, freshly re-stamped (``updated_at`` advancing) at
  least ``replay_min_restamps`` times in a row: a producer re-emitting a frozen reading. A value
  carried forward with ``updated_at`` unchanged is designed (``absent_means``) and never counts.
  Level channels only (counts, buckets and binaries repeat exactly by construction). When the
  samples carry a ``source`` (the row the value was computed from), a run also needs the same
  source throughout; without one, a value at a saturation bound (exactly 0.0 or 1.0) cannot be
  told from a real pinned reading and is not judged as a replay.
* ``constant`` -- level channel, one distinct value over at least ``constant_min_span_sec`` with
  at least ``min_samples`` samples, outside its declared rest range, and freshly written at least
  once (a value only carried forward is "no reading", not a constant reading).
* ``floor_locked`` -- level, ``per_tick`` channel whose minimum over the same span stays above
  the top of its declared rest range: it never reaches rest.

``rest_range`` is ``semantics["rest_range"] = [lo, hi]``. Missing or malformed: every
rest-dependent check (replayed, constant, floor_locked) answers ``cannot_judge`` for that channel;
it is never flagged. One finding per channel per sweep, in priority order
``decayed_to_zero`` > ``replayed`` (with source evidence) > ``constant`` > ``replayed`` >
``floor_locked``. Bad input never raises: the channel is skipped with a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.field.channel_glossary import classify_channel_series

IMMUNE_FINDING_KIND = "immune_finding"
IMMUNE_KINDS = ("constant", "floor_locked", "replayed", "decayed_to_zero")
LEVEL_KIND = "level"
PER_TICK = "per_tick"
# Below this a double is subnormal and loses precision, so ratios between such values are not
# exact: a decay segment stops there (the route value went subnormal on its way to 0.0).
NORMAL_FLOAT_MIN = 1e-290
SATURATION_BOUNDS = (0.0, 1.0)


@dataclass(frozen=True)
class ImmuneConfig:
    # Constant / floor-locked need most of the 24 h window: 20 h span, and at least 600 one-minute
    # samples (10 h of coverage) so two samples 20 h apart cannot pass as a day.
    constant_min_span_sec: float = 20 * 3600.0
    min_samples: int = 600
    # 30 fresh re-stamps in a row at one sample per minute = half an hour of a producer re-writing
    # the identical non-rest value. The live 24 h dry run (2026-10-13) had no healthy level
    # channel repeat a non-rest value with fresh stamps more than 3 times in a row.
    replay_min_restamps: int = 30
    # 5 consecutive exact-geometric changes: the chance a real reading lands within 1e-6 of an
    # integer power of one base five times running is negligible, and the route failure produced
    # one such change per minute for two days.
    decay_min_changes: int = 5
    decay_rel_tol: float = 1e-6
    decay_max_divisor: int = 64
    max_examples: int = 5


@dataclass(frozen=True)
class ImmuneSample:
    t: datetime
    value: float
    updated_at: Optional[datetime] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class ImmuneFinding:
    metric_id: str
    node: str
    channel: str
    kind: str
    window_start: datetime
    window_end: datetime
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"metric_id": self.metric_id, "node": self.node, "channel": self.channel, "kind": self.kind,
                "window_start": self.window_start.isoformat(), "window_end": self.window_end.isoformat(),
                "evidence": dict(self.evidence)}


@dataclass(frozen=True)
class ChannelVerdict:
    """Every check's answer for one channel: ``flagged``, ``clear``, ``cannot_judge:<why>`` or
    ``not_applicable:<why>``. The dry run and the trace read this; only findings are written."""

    metric_id: str
    checks: dict
    liveness: Optional[str] = None
    n_samples: int = 0
    finding: Optional[ImmuneFinding] = None
    skipped: Optional[str] = None


@dataclass(frozen=True)
class ImmuneSweepResult:
    findings: list
    verdicts: list

    @property
    def open_count(self) -> int:
        return len(self.findings)


# --- input normalisation (never raises) --------------------------------------------------------

def _as_dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sample(raw: Any) -> Optional[ImmuneSample]:
    try:
        if isinstance(raw, ImmuneSample):
            t, v, u, s = raw.t, raw.value, raw.updated_at, raw.source
        elif isinstance(raw, Mapping):
            t, v, u, s = raw.get("t"), raw.get("value"), raw.get("updated_at"), raw.get("source")
        elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
            t, v = raw[0], raw[1]
            u = raw[2] if len(raw) > 2 else None
            s = raw[3] if len(raw) > 3 else None
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    td, fv = _as_dt(t), _as_float(v)
    if td is None or fv is None:
        return None
    return ImmuneSample(t=td, value=fv, updated_at=_as_dt(u), source=None if s is None else str(s))


def _semantics(raw: Any) -> dict:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        return dict(raw or ())
    except (TypeError, ValueError):
        return {}


def rest_range_of(semantics: Mapping) -> Optional[tuple[float, float]]:
    """``semantics["rest_range"]`` as ``(lo, hi)``, or None when missing or malformed."""
    rr = semantics.get("rest_range") if isinstance(semantics, Mapping) else None
    if not isinstance(rr, (list, tuple)) or len(rr) != 2:
        return None
    lo, hi = _as_float(rr[0]), _as_float(rr[1])
    if lo is None or hi is None or lo > hi:
        return None
    return lo, hi


def _in_rest(v: float, rest: tuple[float, float]) -> bool:
    return rest[0] <= v <= rest[1]


def _examples(values: Iterable[float], cap: int) -> list:
    out: list = []
    for v in values:
        if len(out) >= cap:
            break
        out.append(v)
    return out


def _fresh(a: ImmuneSample, b: ImmuneSample) -> bool:
    return a.updated_at is not None and b.updated_at is not None and b.updated_at > a.updated_at


# --- the four checks ---------------------------------------------------------------------------

def _decayed(samples: Sequence[ImmuneSample], cfg: ImmuneConfig) -> Optional[dict]:
    """Longest run of strictly falling positive values whose steps are exact powers of one ratio."""
    # Changes only: equal neighbours are "no decay tick between these two samples" (ratio**0).
    pts: list[ImmuneSample] = []
    for s in samples:
        if not pts or s.value != pts[-1].value:
            pts.append(s)
    best: Optional[dict] = None
    seg: list[ImmuneSample] = []

    def judge(seg: list[ImmuneSample]) -> Optional[dict]:
        if len(seg) - 1 < cfg.decay_min_changes:
            return None
        logs = [math.log(b.value / a.value) for a, b in zip(seg, seg[1:])]
        smallest = min(abs(x) for x in logs)
        if smallest <= 0.0:
            return None
        for d in range(1, cfg.decay_max_divisor + 1):
            base = smallest / d
            powers = []
            for x in logs:
                m = round(abs(x) / base)
                if m < 1 or abs(abs(x) - m * base) > cfg.decay_rel_tol * abs(x):
                    break
                powers.append(m)
            else:
                ratio = math.exp(-base)
                return {"changes": len(logs), "ratio": round(ratio, 9), "powers": powers[: cfg.max_examples],
                        "first_value": seg[0].value, "last_value": seg[-1].value,
                        "examples": _examples((p.value for p in seg), cfg.max_examples),
                        "start": seg[0].t, "end": seg[-1].t,
                        "updated_at_moved": any(_fresh(a, b) for a, b in zip(seg, seg[1:]))}
        return None

    for s in pts:
        if s.value > NORMAL_FLOAT_MIN and (not seg or s.value < seg[-1].value):
            seg.append(s)
            continue
        hit = judge(seg)
        if hit and (best is None or hit["changes"] > best["changes"]):
            best = hit
        seg = [s] if s.value > NORMAL_FLOAT_MIN else []
    hit = judge(seg)
    if hit and (best is None or hit["changes"] > best["changes"]):
        best = hit
    return best


def _replayed(samples: Sequence[ImmuneSample], rest: tuple[float, float], cfg: ImmuneConfig) -> Optional[dict]:
    best: Optional[dict] = None
    run: list[ImmuneSample] = []

    def judge(run: list[ImmuneSample]) -> Optional[dict]:
        restamps = len(run) - 1
        if restamps < cfg.replay_min_restamps:
            return None
        sourced = run[0].source is not None
        if not sourced and run[0].value in SATURATION_BOUNDS:
            return None
        return {"restamps": restamps, "value": run[0].value, "source": run[0].source,
                "source_evidence": sourced, "start": run[0].t, "end": run[-1].t,
                "first_updated_at": run[0].updated_at.isoformat() if run[0].updated_at else None,
                "last_updated_at": run[-1].updated_at.isoformat() if run[-1].updated_at else None}

    for s in samples:
        if (run and s.value == run[-1].value and _fresh(run[-1], s) and s.source == run[-1].source
                and not _in_rest(s.value, rest)):
            run.append(s)
            continue
        hit = judge(run)
        if hit and (best is None or hit["restamps"] > best["restamps"]):
            best = hit
        run = [s] if (not _in_rest(s.value, rest) and s.updated_at is not None) else []
    hit = judge(run)
    if hit and (best is None or hit["restamps"] > best["restamps"]):
        best = hit
    return best


def _long_enough(samples: Sequence[ImmuneSample], cfg: ImmuneConfig) -> Optional[str]:
    if len(samples) < cfg.min_samples:
        return f"cannot_judge:samples {len(samples)} < {cfg.min_samples}"
    span = (samples[-1].t - samples[0].t).total_seconds()
    if span < cfg.constant_min_span_sec:
        return f"cannot_judge:span {span / 3600:.1f} h < {cfg.constant_min_span_sec / 3600:.0f} h"
    return None


def _ever_written(samples: Sequence[ImmuneSample]) -> bool:
    """Freshly written at least once in the window (or the series carries no stamps at all)."""
    stamps = [s.updated_at for s in samples]
    if all(u is None for u in stamps):
        return True
    return len({u for u in stamps if u is not None}) > 1


# --- one channel, then the sweep ---------------------------------------------------------------

def inspect_channel(
    node: str,
    channel: str,
    raw_samples: Any,
    raw_semantics: Any,
    *,
    config: ImmuneConfig = ImmuneConfig(),
) -> ChannelVerdict:
    metric_id = f"{node}.{channel}"
    try:
        sem = _semantics(raw_semantics)
        kind = sem.get("value_kind")
        if not sem or kind is None:
            return ChannelVerdict(metric_id, {}, skipped="no semantics")
        if kind == "placeholder":
            return ChannelVerdict(metric_id, {}, skipped="placeholder: its entry already says it measures nothing")
        samples = sorted((s for s in (_sample(r) for r in (raw_samples or ())) if s is not None), key=lambda s: s.t)
        if len(samples) < 2:
            return ChannelVerdict(metric_id, {}, n_samples=len(samples), skipped="fewer than 2 usable samples")
        values = [s.value for s in samples]
        liveness = classify_channel_series(values)
        rest = rest_range_of(sem)
        level = kind == LEVEL_KIND
        checks: dict = {}
        found: dict = {}

        dec = _decayed(samples, config)
        checks["decayed_to_zero"] = "flagged" if dec else "clear"
        if dec:
            found["decayed_to_zero"] = dec

        no_rest = "cannot_judge:no rest_range"
        if not level:
            checks["replayed"] = checks["constant"] = checks["floor_locked"] = f"not_applicable:value_kind {kind}"
        elif rest is None:
            checks["replayed"] = checks["constant"] = checks["floor_locked"] = no_rest
        else:
            rep = _replayed(samples, rest, config)
            checks["replayed"] = "flagged" if rep else "clear"
            if rep:
                found["replayed"] = rep
            short = _long_enough(samples, config)
            distinct = len(set(values))
            if short:
                checks["constant"] = checks["floor_locked"] = short
            elif not _ever_written(samples):
                checks["constant"] = checks["floor_locked"] = "clear:carried forward, never re-written (designed)"
            else:
                if distinct == 1 and not _in_rest(values[0], rest):
                    checks["constant"] = "flagged"
                    found["constant"] = {"value": values[0]}
                else:
                    checks["constant"] = "clear"
                if sem.get("sparsity") != PER_TICK:
                    checks["floor_locked"] = f"not_applicable:sparsity {sem.get('sparsity')}"
                elif min(values) > rest[1]:
                    checks["floor_locked"] = "flagged"
                    found["floor_locked"] = {}
                else:
                    checks["floor_locked"] = "clear"

        finding = _pick(metric_id, node, channel, samples, rest, found, config)
        return ChannelVerdict(metric_id, checks, liveness=liveness, n_samples=len(samples), finding=finding)
    except Exception as exc:  # noqa: BLE001 - one bad channel never stops the sweep
        return ChannelVerdict(metric_id, {}, skipped=f"error {type(exc).__name__}: {str(exc)[:120]}")


def _pick(metric_id, node, channel, samples, rest, found, cfg) -> Optional[ImmuneFinding]:
    values = [s.value for s in samples]
    base = {"n_samples": len(samples), "distinct": len(set(values)), "min": min(values), "max": max(values),
            "rest_range": list(rest) if rest else None}
    rep = found.get("replayed")
    order = ["decayed_to_zero"]
    order += ["replayed", "constant"] if rep and rep["source_evidence"] else ["constant", "replayed"]
    order += ["floor_locked"]
    for kind in order:
        hit = found.get(kind)
        if hit is None:
            continue
        ev = dict(base)
        if kind in ("decayed_to_zero", "replayed"):
            start, end = hit["start"], hit["end"]
            ev.update({k: v for k, v in hit.items() if k not in ("start", "end")})
        else:
            start, end = samples[0].t, samples[-1].t
            ev.update(hit)
            ev["examples"] = _examples(dict.fromkeys(values), cfg.max_examples)
        return ImmuneFinding(metric_id, node, channel, kind, start, end, ev)
    return None


def semantic_channel_names(entries: Iterable[Any]) -> list[str]:
    """Channel names the sweep must read: every glossary entry that carries semantics."""
    return sorted({e.channel for e in entries if getattr(e, "semantics", None)})


def semantic_channels(entries: Iterable[Any], observed: Iterable[tuple[str, str]]) -> dict:
    """``(node, channel) -> semantics`` for the glossary entries that carry semantics.

    Same resolution as ``orion.field.channel_glossary.resolve_channel_entry``: a node-qualified
    entry wins for its node; a bare entry covers that channel on every other observed node. Only
    the resolved entry's semantics count (a node whose resolved entry has none is not judged).
    Qualified entries are included even when unobserved, so the sweep reports them absent."""
    entries = list(entries)
    qualified = {(e.node, e.channel): e for e in entries if e.node}
    bare: dict = {}
    for e in entries:
        if not e.node and e.channel not in bare:
            bare[e.channel] = e
    out: dict = {k: dict(e.semantics) for k, e in qualified.items() if e.semantics}
    for node, channel in observed:
        if (node, channel) in qualified:
            continue
        e = bare.get(channel)
        if e is not None and e.semantics:
            out[(node, channel)] = dict(e.semantics)
    return out


def sweep(
    series: Mapping[tuple[str, str], Any],
    semantics: Mapping[tuple[str, str], Any],
    *,
    config: ImmuneConfig = ImmuneConfig(),
) -> ImmuneSweepResult:
    """Judge every ``(node, channel)`` series that has semantics. A pair with semantics but no
    series is reported as skipped ("absent in the window"), never as a finding."""
    verdicts: list[ChannelVerdict] = []
    findings: list[ImmuneFinding] = []
    try:
        keys = sorted(set(series or {}) | set(semantics or {}), key=lambda k: (str(k[0]), str(k[1])))
    except Exception:  # noqa: BLE001
        return ImmuneSweepResult([], [])
    for key in keys:
        try:
            node, channel = str(key[0]), str(key[1])
        except Exception:  # noqa: BLE001
            continue
        sem = (semantics or {}).get(key)
        if sem is None:
            continue
        raw = (series or {}).get(key)
        if not raw:
            verdicts.append(ChannelVerdict(f"{node}.{channel}", {}, skipped="absent in the window"))
            continue
        v = inspect_channel(node, channel, raw, sem, config=config)
        verdicts.append(v)
        if v.finding is not None:
            findings.append(v.finding)
    return ImmuneSweepResult(findings, verdicts)


__all__ = [
    "IMMUNE_FINDING_KIND",
    "IMMUNE_KINDS",
    "ChannelVerdict",
    "ImmuneConfig",
    "ImmuneFinding",
    "ImmuneSample",
    "ImmuneSweepResult",
    "inspect_channel",
    "rest_range_of",
    "semantic_channel_names",
    "semantic_channels",
    "sweep",
]
