"""One body summary per arc: existing sensors at arc altitude. Pure; the caller passes the
rows that fall inside the arc's interval (patch 3 reads them per closed arc).

Provenance, one line per field (metric gate step 1):

* ``chassis_watts_mean``: ``orion_biometrics_cluster.chassis_watts`` (``observed_at``
  timestamptz). Rest is the idle draw, not 0 (live 10-09 per-arc means 1,093-2,007 W).
* ``cabinet_temp_c_min``/``max``: athena's ``orion_biometrics_summary.measurements
  ->'cabinet_temp_c'`` (``timestamp`` is TEXT ISO, cast via ``day.as_utc``).
* ``ambient_spike_count``: ``cabinet_ambient_spike`` rows.
* ``thermal_refusals``: visual deferrals whose own ``thermal_state`` read ``hot``
  (12 ``deferred_thermal`` attempts in the 30 days to 10-10; 0 on most arcs).

Not built, because they failed the metric gate on live data (see ``ArcBodySummaryV1``):
the cluster ``peak_pressure`` max and cooling switch changes.

No feeling is asserted; the field is called ``body``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from orion.schemas.temporal_self import ArcBodySummaryV1, TemporalSelfEventV1

Row = Mapping[str, Any]


def _f(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def summarize_body(
    cluster_rows: Iterable[Row] = (),
    cabinet_rows: Iterable[Row] = (),
    spike_rows: Iterable[Row] = (),
    deferrals: Iterable[TemporalSelfEventV1] = (),
) -> ArcBodySummaryV1:
    cluster = list(cluster_rows)
    watts = [w for w in (_f(r.get("chassis_watts")) for r in cluster) if w is not None]
    temps = [t for t in (_f(r.get("cabinet_temp_c")) for r in cabinet_rows) if t is not None]
    return ArcBodySummaryV1(
        cluster_sample_count=len(cluster),
        chassis_watts_mean=round(sum(watts) / len(watts), 3) if watts else None,
        cabinet_sample_count=len(temps),
        cabinet_temp_c_min=min(temps) if temps else None,
        cabinet_temp_c_max=max(temps) if temps else None,
        ambient_spike_count=sum(1 for _ in spike_rows),
        thermal_refusals=sum(
            1 for e in deferrals
            if e.source_kind == "visual_deferral" and (e.payload.get("thermal_state") == "hot" or e.verdict == "deferred_thermal")
        ),
    )
