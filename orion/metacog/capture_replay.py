"""orion/metacog/capture_replay.py

Pure analysis for replaying metacog triggers through
``orion.metacog.evidence_map`` (spec 2026-09-24 section C, acceptance checks
4-6). No IO: ``scripts/analysis/replay_metacog_capture.py`` feeds it live rows
read-only from Postgres, and ``orion/metacog/evals/run_capture_eval.py`` feeds
it the committed real-row fixture. Kept in ``orion/`` (not ``scripts/``) so the
eval can import it without inverting the repo's layering.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from orion.metacog.evidence_map import map_trigger, severity_rank

SEVERITIES = ("nominal", "degraded", "critical")

# Spec acceptance thresholds.
SPEARMAN_TARGET = 0.6
# Check 4a applies to transport and telemetry (spec). chat_turn's fired-count
# proxy is reported, not gated: conditions of different weight are not a
# magnitude.
GATED_PROXIES = (
    "rpc_health:p95_over_threshold",
    "rpc_health:timeout_count",
    "bus_synaptic:error_over_threshold",
    "telemetry:recon_over_threshold",
)
DISTINCT_DENSITY_PER_DAY_TARGET = 10


# ---------------------------------------------------------------------------
# stats (stdlib only)
# ---------------------------------------------------------------------------


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rho with average ranks for ties. None if undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def tie_limited_ceiling(sev_ranks: list[int], proxies: list[float]) -> float | None:
    """The best Spearman rho ANY severity assignment with the same class sizes
    could reach against `proxies`: hand out the same multiset of severities in
    proxy order. A 3-level ordinal against a continuous proxy is capped well
    below 1 by ties (e.g. mostly-nominal kinds), so rho alone conflates "the
    mapper mis-orders rows" with "the classes are unbalanced". rho == ceiling
    means severity is perfectly monotone in the proxy."""
    if len(sev_ranks) != len(proxies) or len(proxies) < 3:
        return None
    ordered_sev = sorted(sev_ranks)
    order = sorted(range(len(proxies)), key=lambda i: proxies[i])
    ideal = [0] * len(proxies)
    for rank_pos, idx in enumerate(order):
        ideal[idx] = ordered_sev[rank_pos]
    return spearman([float(x) for x in ideal], proxies)


# Eval tolerance for "monotone": rho must reach this fraction of its
# tie-limited ceiling. Not 1.0 because the rpc_health thin-sample cap (spec
# A3.2) deliberately reorders a few latency-only rows.
CEILING_FRACTION = 0.95


# ---------------------------------------------------------------------------
# raw magnitude proxies (independent of evidence_map's own magnitude)
# ---------------------------------------------------------------------------


def _f(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def proxy_for(kind: str, up: dict[str, Any]) -> tuple[str, float] | None:
    """(proxy_name, raw value) straight from upstream, or None."""
    if not isinstance(up, dict):
        return None
    src = up.get("evidence_source")
    if kind == "transport" and src == "rpc_health_snapshot":
        t = _f(up.get("timeout_count")) or 0.0
        if t > 0:
            return ("rpc_health:timeout_count", t)
        p95, thr = _f(up.get("success_latency_ms_p95")), _f(up.get("latency_p95_threshold_ms"))
        if p95 is not None and thr:
            return ("rpc_health:p95_over_threshold", p95 / thr)
        return None
    if kind == "transport" and src == "bus_synaptic_prediction_error":
        e, thr = _f(up.get("error")), _f(up.get("error_threshold"))
        if e is not None and thr:
            return ("bus_synaptic:error_over_threshold", e / thr)
        return None
    if kind == "telemetry_anomaly":
        loss, thr = _f(up.get("recon_loss")), _f(up.get("threshold"))
        if loss is not None and thr:
            return ("telemetry:recon_over_threshold", loss / thr)
        return None
    if kind == "chat_turn":
        fired = up.get("fired_conditions")
        if isinstance(fired, list):
            return ("chat_turn:fired_condition_count", float(len(fired)))
    return None


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def analyze(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    per_kind_sev: dict[str, Counter] = defaultdict(Counter)
    per_kind_mags: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    proxy_pairs: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    distinct_new_per_day: dict[str, set] = defaultdict(set)
    distinct_old_per_day: dict[str, set] = defaultdict(set)
    zero_without_zero_mag = 0
    no_evidence = Counter()
    old_vs_new: dict[str, Counter] = defaultdict(Counter)
    old_proxy_pairs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    zen_in_fallback = 0
    rows_per_day: Counter = Counter()
    n = 0

    for row in rows:
        n += 1
        kind = str(row.get("trigger_kind") or "")
        up = row.get("upstream")
        m = map_trigger(kind, row.get("reason"), up)
        per_kind_sev[kind][m.severity] += 1
        per_kind_mags[kind][m.severity].append(m.magnitude)
        day = str(row.get("timestamp") or "")[:10] or "unknown"
        distinct_new_per_day[day].add(round(m.magnitude, 4))
        rows_per_day[day] += 1
        if m.causal_density["score"] == 0.0 and m.magnitude != 0.0:
            zero_without_zero_mag += 1
        if m.density_rationale.startswith("no_evidence"):
            no_evidence[kind] += 1
        if "zen" in m.summary_fallback.lower():
            zen_in_fallback += 1
        px = proxy_for(kind, up if isinstance(up, dict) else {})
        if px is not None:
            proxy_pairs[px[0]].append((severity_rank(m.severity), px[1], m.magnitude))
        old_sev = row.get("old_severity")
        if old_sev:
            old_vs_new[kind][(old_sev, m.severity)] += 1
            if px is not None:
                old_proxy_pairs[px[0]].append((severity_rank(old_sev), px[1]))
        old_score = row.get("old_density_score")
        if old_score is not None:
            distinct_old_per_day[day].add(round(float(old_score), 4))
        if (len(sample_rows[kind]) < 3 and m.severity != "nominal") or not sample_rows[kind]:
            sample_rows[kind].append(
                {
                    "reason": str(row.get("reason") or "")[:120],
                    "severity": m.severity,
                    "magnitude": m.magnitude,
                    "evidence": m.evidence[:3],
                    "touches": m.touches[:4],
                    "old_severity": old_sev,
                }
            )

    # Check 4a: Spearman per proxy.
    rho = {}
    rho_old = {}
    for name, pairs in proxy_pairs.items():
        rho[name] = {
            "n": len(pairs),
            "rho": spearman([p[0] for p in pairs], [p[1] for p in pairs]),
            "ceiling": tie_limited_ceiling([p[0] for p in pairs], [p[1] for p in pairs]),
        }
    for name, pairs in old_proxy_pairs.items():
        rho_old[name] = {"n": len(pairs), "rho": spearman([p[0] for p in pairs], [p[1] for p in pairs])}

    # Check 4b: no nominal row above the median critical row, per kind, on
    # both the mapped magnitude and the raw proxy.
    check4b = {}
    for kind, by_sev in per_kind_mags.items():
        crit = by_sev.get("critical") or []
        nom = by_sev.get("nominal") or []
        if not crit:
            check4b[kind] = {"status": "n/a (no critical rows)"}
            continue
        med = statistics.median(crit)
        violators = sum(1 for x in nom if x > med)
        check4b[kind] = {
            "median_critical_magnitude": round(med, 4),
            "max_nominal_magnitude": round(max(nom), 4) if nom else None,
            "violators": violators,
            "status": "PASS" if violators == 0 else "FAIL",
        }
    check4b_proxy = {}
    for name, pairs in proxy_pairs.items():
        crit = [p[1] for p in pairs if p[0] == 2]
        nom = [p[1] for p in pairs if p[0] == 0]
        if not crit:
            check4b_proxy[name] = {"status": "n/a (no critical rows)"}
            continue
        med = statistics.median(crit)
        violators = sum(1 for x in nom if x > med)
        check4b_proxy[name] = {
            "median_critical_proxy": round(med, 4),
            "violators": violators,
            "status": "PASS" if violators == 0 else "FAIL",
        }

    # Check 5: distinct density scores per day.
    per_day = {
        day: {
            "new_distinct": len(distinct_new_per_day[day]),
            "old_distinct": len(distinct_old_per_day.get(day, set())) if distinct_old_per_day else None,
        }
        for day in sorted(distinct_new_per_day)
    }

    return {
        "rows": n,
        "rows_per_day": dict(rows_per_day),
        "severity_by_kind": {k: {s: c.get(s, 0) for s in SEVERITIES} for k, c in sorted(per_kind_sev.items())},
        "no_evidence_by_kind": dict(no_evidence),
        "spearman_new": rho,
        "spearman_old": rho_old,
        "check4b_magnitude": check4b,
        "check4b_proxy": check4b_proxy,
        "density_distinct_per_day": per_day,
        "zero_score_with_nonzero_magnitude": zero_without_zero_mag,
        "zen_in_fallback_summaries": zen_in_fallback,
        "old_vs_new": {
            k: {f"{a}->{b}": c for (a, b), c in sorted(v.items())} for k, v in sorted(old_vs_new.items())
        },
        "samples": dict(sample_rows),
    }


def _fmt_rho(v: float | None) -> str:
    return "undefined" if v is None else f"{v:.3f}"


def render_report(res: dict[str, Any], *, source: str) -> str:
    L: list[str] = []
    L.append("# Metacog capture replay (evidence_map)\n")
    L.append(f"Source: {source}. Rows replayed: **{res['rows']}**. Read-only; nothing was written.\n")

    L.append("## Severity distribution per kind (new, event-derived)\n")
    L.append("| kind | nominal | degraded | critical | no_evidence |")
    L.append("|---|---|---|---|---|")
    for k, d in res["severity_by_kind"].items():
        L.append(f"| {k} | {d['nominal']} | {d['degraded']} | {d['critical']} | {res['no_evidence_by_kind'].get(k, 0)} |")

    L.append("\n## Check 4a: Spearman rho, severity vs raw magnitude proxy (target >= 0.6)\n")
    L.append("| proxy | n | rho NEW | tie-limited ceiling | rho OLD (published severity) |")
    L.append("|---|---|---|---|---|")
    for name, d in sorted(res["spearman_new"].items()):
        old = res["spearman_old"].get(name, {})
        L.append(
            f"| {name} | {d['n']} | {_fmt_rho(d['rho'])} | {_fmt_rho(d.get('ceiling'))} | "
            f"{_fmt_rho(old.get('rho'))} (n={old.get('n', 0)}) |"
        )
    L.append(
        "\nNote: telemetry and bus_synaptic severities are banded functions of exactly this proxy, "
        "so their rho is high by construction (a regression guard, not independent validation). "
        "`rpc_health:*` rows split timeout rows from latency-only rows because the mapper leads with timeouts. "
        "The tie-limited ceiling is the best rho any 3-level severity with the same class sizes could reach; "
        "rho == ceiling means severity is perfectly monotone in the proxy. chat_turn is reported, not gated."
    )

    L.append("\n## Check 4b: no nominal row above the median critical row of its kind\n")
    L.append("On mapped magnitude:\n")
    L.append("| kind | median critical | max nominal | violators | status |")
    L.append("|---|---|---|---|---|")
    for k, d in sorted(res["check4b_magnitude"].items()):
        L.append(
            f"| {k} | {d.get('median_critical_magnitude', '-')} | {d.get('max_nominal_magnitude', '-')} | "
            f"{d.get('violators', '-')} | {d['status']} |"
        )
    L.append("\nOn the raw proxy (the stronger version):\n")
    L.append("| proxy | median critical proxy | violators | status |")
    L.append("|---|---|---|---|")
    for k, d in sorted(res["check4b_proxy"].items()):
        L.append(f"| {k} | {d.get('median_critical_proxy', '-')} | {d.get('violators', '-')} | {d['status']} |")

    L.append("\n## Check 5: distinct causal_density scores per day (target > 10)\n")
    L.append("| day | NEW distinct | OLD distinct (published) |")
    L.append("|---|---|---|")
    for day, d in res["density_distinct_per_day"].items():
        L.append(f"| {day} | {d['new_distinct']} | {d['old_distinct'] if d['old_distinct'] is not None else '-'} |")
    L.append(
        f"\nRows where score is 0 but magnitude is not: **{res['zero_score_with_nonzero_magnitude']}** (target 0)."
    )

    L.append("\n## Check 6 (deterministic half only)\n")
    L.append(
        f"Deterministic fallback summaries containing 'zen': **{res['zen_in_fallback_summaries']}**. "
        "LLM-authored summaries can only be checked after deploy: **UNVERIFIED** until live rows exist."
    )

    if res["old_vs_new"]:
        L.append("\n## Old published severity -> new severity (joined on correlation_id)\n")
        for k, d in res["old_vs_new"].items():
            L.append(f"- **{k}**: " + ", ".join(f"{t}: {c}" for t, c in d.items()))

    L.append("\n## Examples\n")
    for k, rows in sorted(res["samples"].items()):
        for r in rows:
            L.append(
                f"- `{k}` {r['severity']} ({r['magnitude']:.3f}; old={r['old_severity']}): "
                f"{'; '.join(r['evidence'])} -- touches {', '.join(r['touches'])}"
            )
    return "\n".join(L) + "\n"




def acceptance_failures(
    res: dict[str, Any], *, min_rows_per_day: int = 100, rho_mode: str = "spec"
) -> list[str]:
    """Checks 4-6 as hard failures. Days with fewer than `min_rows_per_day`
    replayed rows are skipped for check 5 (a partial day, e.g. today).

    rho_mode="spec": rho >= 0.6 (the spec's number; use on the live
    population). rho_mode="ceiling": rho >= CEILING_FRACTION x its
    tie-limited ceiling (use on a stratified fixture, whose class balance is
    not the live one)."""
    fails: list[str] = []
    for name in GATED_PROXIES:
        d = res["spearman_new"].get(name)
        if not d or d["n"] < 3:
            continue
        if d["rho"] is None:
            fails.append(f"check4a {name}: rho undefined (n={d['n']})")
        elif rho_mode == "ceiling":
            ceiling = d.get("ceiling")
            if ceiling is None or d["rho"] < CEILING_FRACTION * ceiling:
                fails.append(f"check4a {name}: rho={d['rho']:.3f} < {CEILING_FRACTION} x ceiling {ceiling}")
        elif d["rho"] < SPEARMAN_TARGET:
            fails.append(f"check4a {name}: rho={d['rho']:.3f} < {SPEARMAN_TARGET} (n={d['n']})")
    for kind, d in res["check4b_magnitude"].items():
        if d.get("status") == "FAIL":
            fails.append(f"check4b magnitude {kind}: {d['violators']} nominal rows above median critical")
    for name, d in res["check4b_proxy"].items():
        if d.get("status") == "FAIL":
            fails.append(f"check4b proxy {name}: {d['violators']} nominal rows above median critical")
    for day, d in res["density_distinct_per_day"].items():
        if res["rows_per_day"].get(day, 0) < min_rows_per_day:
            continue
        if d["new_distinct"] <= DISTINCT_DENSITY_PER_DAY_TARGET:
            fails.append(f"check5 {day}: only {d['new_distinct']} distinct density scores")
    if res["zero_score_with_nonzero_magnitude"]:
        fails.append("check5: causal_density.score is 0 while magnitude is not")
    if res["zen_in_fallback_summaries"]:
        fails.append(f"check6: {res['zen_in_fallback_summaries']} deterministic summaries mention zen")
    return fails
