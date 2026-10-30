"""The machine's overhead factor: how many watts at the wall each watt of GPU/CPU load costs.

Every cost number Orion reads means one thing: WHOLE-MACHINE power at the wall above rest, fans and
power-supply loss included. The per-act meter (orion/energy/act_energy.py, PR #2992) gets that by
splitting each second's chassis watts above rest over the load above idle. The component views
(a card's watts, a container's CPU watts, a GPU-only run meter) do not see the chassis. This factor
turns a component's watts above idle into an ESTIMATE of the wall watts above rest it drives:

    chassis above rest  ~=  k x (GPU above idle + CPU above idle)

THEORY (same as act_energy.py): fans, PSU conversion loss, memory and VRMs draw in proportion to
the component load that makes the heat. The meter applies it per second (the ratio); this module
fits it per host over hourly means of the SAME readings (``chassis_watts``, ``gpu_watts_total``,
``cpu_watts_total``, ``cpu_idle_watts`` in orion_biometrics_summary) against the SAME pinned rest
(config/physiology/rest_body.yaml), as a least-squares slope through zero.

NOT NEW SIGNAL: k is a derived view of the chassis measurement the meter already splits. A
component's ``x k`` figure is an estimate of its share of that chassis draw, never a reading.

When a host's fit is not credible (too few hours, R^2 under MIN_R2, k under 1), the factor is
#2992's circe fit, said as such (``fitted=False`` and the basis names it). Pure arithmetic: the
caller reads the rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Mapping, Optional

#: PR #2992's live fit: circe 2026-10-16..30, 339 hourly means, rest 541 W, CPU idle 100 W:
#: chassis above rest = 2.69 x (GPU above idle + CPU above idle), R^2 0.79.
FALLBACK_FACTOR = 2.69
FALLBACK_SOURCE = "PR #2992 fit on circe 2026-10-16..30 (339 hourly means, R^2 0.79)"
#: Trailing history the per-host fit reads (the #2992 fit used two weeks).
FIT_WINDOW = timedelta(days=14)
#: Hours with some load above idle needed before a fit is used.
MIN_HOURS = 48
#: Below this the line does not describe the host: use the fallback.
MIN_R2 = 0.5
#: The wall cannot draw less than the components inside it: a k under 1 is a broken fit.
MIN_FACTOR = 1.0


@dataclass(frozen=True)
class Overhead:
    host: str
    factor: float
    basis: str
    fitted: bool
    r2: Optional[float] = None
    hours: int = 0

    def wall(self, component_above_idle_w: Optional[float]) -> Optional[float]:
        """Estimated wall watts above rest that a component's above-idle watts drive."""
        if component_above_idle_w is None:
            return None
        return max(0.0, float(component_above_idle_w)) * self.factor


def fallback(host: str, why: str) -> Overhead:
    return Overhead(host=host, factor=FALLBACK_FACTOR, fitted=False,
                    basis=f"{FALLBACK_SOURCE}; not fitted on {host}: {why}")


def _num(v: object) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fit_overhead(host: str, hourly: Iterable[Mapping], *, rest_w: Optional[float], gpu_idle_w: Optional[float],
                 window_words: str = "") -> Overhead:
    """Least-squares k through zero over hourly means.

    ``hourly`` rows: ``ch`` (chassis W), ``gpu`` (all cards W), ``cpu`` (CPU package W), ``cpu_idle``
    (the row's fitted CPU idle; may be None for hours before it was published, then the host's own
    median of the hours that have it, the setting is per host). y = max(0, ch - rest_w);
    x = max(0, gpu - gpu_idle_w) + max(0, cpu - cpu_idle)."""
    if rest_w is None:
        return fallback(host, "no pinned rest")
    if gpu_idle_w is None:
        return fallback(host, "no GPU rest over the pinned window")
    rows = [(_num(r.get("ch")), _num(r.get("gpu")), _num(r.get("cpu")), _num(r.get("cpu_idle"))) for r in hourly]
    rows = [r for r in rows if None not in r[:3]]
    idles = sorted(r[3] for r in rows if r[3] is not None)
    if not idles:
        return fallback(host, "no cpu_idle_watts published")
    cpu_idle_host = idles[len(idles) // 2]
    xs, ys = [], []
    for ch, gpu, cpu, ci in rows:
        xs.append(max(0.0, gpu - gpu_idle_w) + max(0.0, cpu - (ci if ci is not None else cpu_idle_host)))
        ys.append(max(0.0, ch - rest_w))
    loaded = sum(1 for x in xs if x > 0.0)
    if loaded < MIN_HOURS:
        return fallback(host, f"{loaded} loaded hours, need {MIN_HOURS}")
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    k = sxy / sxx
    mean_y = sum(ys) / len(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - k * x) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    if r2 is None or r2 < MIN_R2:
        return fallback(host, f"fit R^2 {r2 if r2 is None else round(r2, 2)} under {MIN_R2}")
    if k < MIN_FACTOR:
        return fallback(host, f"fit k {k:.2f} under {MIN_FACTOR:g}")
    span = f" {window_words}" if window_words else ""
    return Overhead(host=host, factor=round(k, 3), fitted=True, r2=round(r2, 3), hours=len(xs),
                    basis=(f"fitted on {host}{span}: chassis above pinned rest {rest_w:g} W = {k:.2f} x (GPU above "
                           f"{gpu_idle_w:.0f} W + CPU above {cpu_idle_host:.0f} W idle), {len(xs)} hourly means, "
                           f"R^2 {r2:.2f}; same readings and rest as the per-act meter (PR #2992)"))


__all__ = ["FALLBACK_FACTOR", "FALLBACK_SOURCE", "FIT_WINDOW", "MIN_FACTOR", "MIN_HOURS", "MIN_R2", "Overhead",
           "fallback", "fit_overhead"]
