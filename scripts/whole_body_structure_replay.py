#!/usr/bin/env python3
"""Causal geometry Patch 2 replay (#3290): whole-body structure features + their metric gate.

Read-only. Loads 5-minute buckets from live Postgres:

* body:  orion_biometrics_summary.pressures (node athena), mean per bucket
* bus:   bus_traffic_rollup_5m.channel_counts, log1p(count); a partial bucket is missing
* mind:  attention salience traces, acts per actor (grammar_acts), unified turns
         (cockpit_turn_sighting), curiosity completions -- counts per bucket
* world: Juniper's chat messages, Hub attention beats, vision events per camera and kind,
         cooling watts -- counts or means per bucket. Counts only, never content.

Turns levels into time-of-day surprise residuals, then computes the Layer 2 features hourly over
rolling 24 h windows, each against a shifted null, and applies the spec's gate:

1. beats its shifted null in >= 50% of windows (observed beyond every null draw)
2. rest reading: on quiet nights it reaches within 1 null SD of the null
3. leading modes recognisable day to day (matched loading similarity >= 0.7)
4. |Spearman| > 0.9 between features -> the copy is named and dropped
5. Layer 1: the top 50 lagged links hold at least one link that is not body (cabinet) hardware

It also builds the organ series for the integration rung (#3296): the stable modes' scores. It
then runs a first field-demon read on them (orion/substrate/demon.py): the horizon, and the
full cut's cost against its null.

Writes results.json and report.md to --out (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orion.substrate import demon  # noqa: E402
from orion.substrate import whole_body_structure as wbs  # noqa: E402
from orion.substrate.causal_geometry_engine import (  # noqa: E402
    DEFAULT_POSTGRES_URI,
    compute_pairwise_results,
)

BUCKET = 300
WINDOW = wbs.BUCKETS_PER_DAY
STEP = wbs.BUCKETS_PER_HOUR
N_NULL = 20
NIGHT_UTC = (7, 13)  # 00:00-06:00 Mountain (MST, UTC-7)
TOP_MODES = 8
STABLE_SIMILARITY = 0.7
PREVIEW_MODES = 6
MAX_ORGANS = 8


# --- load ---------------------------------------------------------------------------------


def _bucket(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() // BUCKET)


def load(uri: str, since: datetime) -> tuple[np.ndarray, list[str], dict[str, str], np.ndarray, dict]:
    """Return (values (T, n), names, group per name, bucket index (T,), source row counts)."""
    import psycopg2

    sums: dict[str, dict[int, float]] = {}
    cnts: dict[str, dict[int, int]] = {}
    group: dict[str, str] = {}
    counted: set[str] = set()  # count channels: an empty bucket is 0, not missing
    rows_seen: dict[str, int] = {}

    def add(name: str, grp: str, b: int, v: float, *, count: bool = False) -> None:
        group[name] = grp
        if count:
            counted.add(name)
        sums.setdefault(name, {})
        cnts.setdefault(name, {})
        sums[name][b] = sums[name].get(b, 0.0) + v
        cnts[name][b] = cnts[name].get(b, 0) + 1

    conn = psycopg2.connect(uri)
    bus_ok: set[int] = set()
    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT timestamp::timestamptz, pressures FROM orion_biometrics_summary "
            "WHERE node = 'athena' AND timestamp::timestamptz >= %s", (since,))
        rows = cur.fetchall()
        rows_seen["orion_biometrics_summary"] = len(rows)
        for ts, pressures in rows:
            for k, v in (pressures or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    add(f"body:{k}", "body", _bucket(ts), float(v))

        cur.execute(
            "SELECT bucket_start, channel_counts, partial FROM bus_traffic_rollup_5m "
            "WHERE bucket_start >= %s", (since,))
        rows = cur.fetchall()
        rows_seen["bus_traffic_rollup_5m"] = len(rows)
        bus_rows = []
        for ts, counts, partial in rows:
            if partial:
                continue
            b = _bucket(ts)
            bus_ok.add(b)
            bus_rows.append((b, counts or {}))
        bus_names = sorted({k for _, c in bus_rows for k in c})
        for b, c in bus_rows:
            for k in bus_names:
                # Body telemetry that rides the bus is body, not mind (review 2026-11-24).
                grp = "body" if k.startswith("orion:biometrics") else "mind"
                add(f"bus:{k}", grp, b, float(np.log1p(c.get(k, 0))))

        def counts_query(sql: str, label: str, grp: str, name_fn) -> None:
            cur.execute(sql, (since,))
            rows = cur.fetchall()
            rows_seen[label] = len(rows)
            for row in rows:
                add(name_fn(row), grp, _bucket(row[0]), 1.0, count=True)

        counts_query("SELECT created_at FROM attention_salience_trace WHERE created_at >= %s",
                     "attention_salience_trace", "mind", lambda r: "mind:salience_traces")
        counts_query("SELECT opened_at, actor FROM grammar_acts WHERE opened_at >= %s",
                     "grammar_acts", "mind", lambda r: f"mind:act:{r[1]}")
        counts_query("SELECT min(ts), correlation_id FROM cockpit_turn_sighting WHERE ts >= %s "
                     "GROUP BY correlation_id", "cockpit_turn_sighting", "mind", lambda r: "mind:turns")
        counts_query("SELECT completed_at FROM curiosity_run_outcomes WHERE completed_at >= %s",
                     "curiosity_run_outcomes", "mind", lambda r: "mind:curiosity_runs")
        counts_query("SELECT timestamp FROM chat_message WHERE role = 'user' AND timestamp >= %s",
                     "chat_message(user)", "world", lambda r: "world:juniper_messages")
        counts_query("SELECT created_at, stream_id, event_type FROM vision_events WHERE created_at >= %s",
                     "vision_events", "world", lambda r: f"world:vision:{r[1]}:{r[2]}")

        cur.execute("SELECT minute, beats FROM hub_attention_minute WHERE minute >= %s", (since,))
        rows = cur.fetchall()
        rows_seen["hub_attention_minute"] = len(rows)
        for ts, beats in rows:
            add("world:hub_attention_beats", "world", _bucket(ts), float(beats or 0), count=True)

        cur.execute("SELECT ts, cooling_watts FROM home_cooling_sample WHERE ts >= %s "
                    "AND cooling_watts IS NOT NULL AND NOT coalesce(stale, false)", (since,))
        rows = cur.fetchall()
        rows_seen["home_cooling_sample"] = len(rows)
        for ts, w in rows:
            add("world:cooling_watts", "world", _bucket(ts), float(w))
    finally:
        conn.close()

    b0 = _bucket(since)
    b1 = _bucket(datetime.now(timezone.utc)) - 1  # last closed bucket
    idx = np.arange(b0, b1 + 1)
    names = sorted(sums)
    x = np.full((len(idx), len(names)), np.nan)
    for j, name in enumerate(names):
        s, c = sums[name], cnts[name]
        is_count = name in counted
        col = np.zeros(len(idx)) if is_count else np.full(len(idx), np.nan)
        for b, v in s.items():
            if b0 <= b <= b1:
                col[b - b0] = v if is_count else v / c[b]
        if name.startswith("bus:"):
            missing = np.array([b not in bus_ok for b in idx])
            col[missing] = np.nan
        x[:, j] = col
    return x, names, group, idx, rows_seen


# --- features over windows ----------------------------------------------------------------

FEATURES = ("coupling", "modes_effective", "leading_share", "redundancy_synergy", "body_mind", "structure_drift")
# Direction in which a feature reads "more structure than chance".
HIGHER_IS_STRUCTURE = {"coupling": True, "modes_effective": False, "leading_share": True,
                       "redundancy_synergy": None, "body_mind": True, "structure_drift": None}


def _features(x: np.ndarray, groups: dict) -> dict:
    f = wbs.window_features(x, groups=groups)
    return {"coupling": f.coupling, "modes_effective": f.modes_effective,
            "leading_share": f.leading_share, "redundancy_synergy": f.redundancy_synergy,
            "body_mind": f.body_mind}


def _verdict(observed: float, null: np.ndarray, higher) -> tuple[float, float, bool]:
    if higher is None:  # two-sided: outside the whole null range
        z, _ = wbs.z_and_p_real(observed, null)
        beats = observed > null.max() or observed < null.min()
        p = float(np.mean(np.abs(null - null.mean()) < abs(observed - null.mean())))
        return z, p, bool(beats)
    z, p = wbs.z_and_p_real(observed, null, higher_is_structure=higher)
    if not higher:
        z = -z  # positive z always means "more structure"
    return z, p, p == 1.0


def score_windows(x: np.ndarray, starts: list[int], length: int, groups: dict, rng, *, drift: bool = True) -> list[dict]:
    """Every feature of each window, with its null (N_NULL shifted draws) and verdict."""
    out = []
    prev_r = None
    for s in starts:
        win = x[s:s + length]
        obs = _features(win, groups)
        r = wbs.correlation(win)
        obs["structure_drift"] = wbs.structure_drift(prev_r, r) if drift and prev_r is not None else None
        prev_r = r
        null = {k: [] for k in FEATURES}
        for _ in range(N_NULL):
            st, gap = wbs.null_starts(len(x) - STEP, length, x.shape[1], rng=rng)
            nw = wbs.shifted_null_window(x, length, st)
            for k, v in _features(nw, groups).items():
                null[k].append(v)
            null["structure_drift"].append(
                wbs.structure_drift(wbs.correlation(nw), wbs.correlation(wbs.shifted_null_window(x, length, st + STEP))))
        row = {"start": s, "null_gap": gap}
        for k in FEATURES:
            if obs.get(k) is None:
                continue
            z, p, beats = _verdict(obs[k], np.array(null[k]), HIGHER_IS_STRUCTURE[k])
            row[k] = {"obs": obs[k], "null_mean": float(np.mean(null[k])), "null_sd": float(np.std(null[k], ddof=1)),
                      "z": z, "p_real": p, "beats": beats}
        out.append(row)
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    for k in FEATURES:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        out[k] = {
            "windows": len(vals),
            "beats_share": float(np.mean([v["beats"] for v in vals])),
            "median_z": float(np.median([v["z"] for v in vals])),
            "median_p_real": float(np.median([v["p_real"] for v in vals])),
            "obs_median": float(np.median([v["obs"] for v in vals])),
            "null_median": float(np.median([v["null_mean"] for v in vals])),
        }
    return out


def spearman_table(rows: list[dict]) -> dict:
    from scipy.stats import spearmanr

    ks = [k for k in FEATURES if all(k in r for r in rows[1:])]
    table = {}
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            pairs = [(r[a]["z"], r[b]["z"]) for r in rows if a in r and b in r]
            if len(pairs) > 5:
                rho = spearmanr(*zip(*pairs)).statistic
                table[f"{a}~{b}"] = float(rho)
    return table


# --- modes, organs, Layer 1 ---------------------------------------------------------------


def mode_names(vec: np.ndarray, names: list[str], top: int = 4) -> list[str]:
    order = np.argsort(np.abs(vec))[::-1][:top]
    return [f"{names[i]}({vec[i]:+.2f})" for i in order]


def daily_stability(x: np.ndarray, day_starts: list[int]) -> dict:
    vecs = []
    for s in day_starts:
        _, v = wbs.top_modes(wbs.correlation(x[s:s + WINDOW]), TOP_MODES)
        vecs.append(v)
    sims = np.array([wbs.match_modes(vecs[i - 1], vecs[i]) for i in range(1, len(vecs))])
    return {
        "days": len(vecs),
        "median_similarity_by_rank": [float(v) for v in np.median(sims, axis=0)] if len(sims) else [],
        "min_similarity_by_rank": [float(v) for v in sims.min(axis=0)] if len(sims) else [],
    }


def organs(x: np.ndarray, names: list[str], stable_by_rank: list[float]) -> tuple[np.ndarray, list[dict], bool]:
    """Organ series: residual scores on the full-period modes that stay recognisable day to day.

    An organ is a mode whose median day-to-day matched loading similarity is >= 0.7 (the spec's
    stability bar). With fewer than 2 such modes there are no organs to cut between; the top
    ``PREVIEW_MODES`` modes are then returned as a labelled preview (third value False), so the
    demon read still has something to run on and nobody mistakes it for organs.
    """
    k = 0
    while k < len(stable_by_rank) and stable_by_rank[k] >= STABLE_SIMILARITY:
        k += 1
    stable = k >= 2
    if not stable:
        k = PREVIEW_MODES
    r = wbs.correlation(x)
    shares, vecs = wbs.top_modes(r, k)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    z = np.where(np.isfinite(x), (x - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    info = [{"rank": i + 1, "share": float(shares[i]), "top_channels": mode_names(vecs[:, i], names)} for i in range(k)]
    return z @ vecs, info, stable


def rank_shares_vs_null(x: np.ndarray, starts: list[int], rng) -> list[float]:
    """Share of windows in which the rank-i eigenvalue beats every null draw at the same rank."""
    wins = np.zeros(MAX_ORGANS)
    for s in starts:
        obs, _ = wbs.top_modes(wbs.correlation(x[s:s + WINDOW]), MAX_ORGANS)
        null = np.array([wbs.top_modes(wbs.correlation(wbs.shifted_null_window(
            x, WINDOW, wbs.null_starts(len(x), WINDOW, x.shape[1], rng=rng)[0])), MAX_ORGANS)[0]
            for _ in range(N_NULL)])
        wins += obs > null.max(axis=0)
    return [float(v) for v in wins / len(starts)]


def layer1_top50(x: np.ndarray, idx: np.ndarray, names: list[str], group: dict) -> dict:
    buckets = {n: {int(idx[t]): float(x[t, j]) for t in np.flatnonzero(np.isfinite(x[:, j]))}
               for j, n in enumerate(names)}
    results = compute_pairwise_results(buckets, n_surrogates=200)
    scored = [p for p in results if p.winner is not None and p.p_value is not None]
    scored.sort(key=lambda p: (p.p_value, -abs(p.winner.r)))
    top = scored[:50]
    non_body = [p for p in top if group[p.channel_a] != "body" or group[p.channel_b] != "body"]
    return {
        "pairs_scored": len(scored),
        "top50_non_body": len(non_body),
        "top10": [{"a": p.channel_a, "b": p.channel_b, "r": round(p.winner.r, 3),
                   "lag_min": p.winner.lag_sec // 60, "p": p.p_value} for p in top[:10]],
        "top_non_body": [{"a": p.channel_a, "b": p.channel_b, "r": round(p.winner.r, 3),
                          "lag_min": p.winner.lag_sec // 60, "p": p.p_value} for p in non_body[:10]],
    }


LEVEL_CHANNELS = ("world:cooling_watts",)


def _active(raw_col: np.ndarray, name: str) -> bool:
    """Spec rule: a count channel must be nonzero in >= 90% of covered buckets. Levels pass."""
    if name.startswith("body:") or name in LEVEL_CHANNELS:
        return True
    covered = raw_col[np.isfinite(raw_col)]
    return covered.size > 0 and float(np.mean(covered != 0)) >= wbs.MIN_PRESENCE


def _merge(x: np.ndarray, names: list[str], group: dict) -> tuple[np.ndarray, list[str], dict, dict]:
    groups = wbs.merge_copies(x)
    merged = wbs.merged_series(x, groups)
    new_names, new_group, merged_sets = [], dict(group), []
    for g in groups:
        name = names[g[0]] if len(g) == 1 else f"{names[g[0]]}(+{len(g) - 1} copies)"
        new_names.append(name)
        new_group[name] = group[names[g[0]]]
        if len(g) > 1:
            merged_sets.append([names[i] for i in g])
    return merged, new_names, new_group, {"before": len(names), "after": len(groups), "groups": merged_sets}


# --- main ---------------------------------------------------------------------------------


def run_block(label: str, x: np.ndarray, idx: np.ndarray, names: list[str], group: dict, rng,
              raw_quiet: np.ndarray) -> dict:
    """``raw_quiet`` is Juniper's messages + Hub attention beats per row, from the raw columns
    (both fail the activity rule, so they are never among ``names``)."""
    groups = {"body": [i for i, n in enumerate(names) if group[n] == "body"],
              "mind": [i for i, n in enumerate(names) if group[n] == "mind"]}
    starts = list(range(0, len(x) - WINDOW - STEP + 1, STEP))
    t0 = time.time()
    rows = score_windows(x, starts, WINDOW, groups, rng)
    # Rest: 6 h windows over 00:00-06:00 Mountain, quiet = no Juniper messages, no Hub attention.
    hour = (idx // wbs.BUCKETS_PER_HOUR) % 24
    night_starts = [int(t) for t in np.flatnonzero((hour == NIGHT_UTC[0]) & (idx % wbs.BUCKETS_PER_HOUR == 0))
                    if t + 72 <= len(x)]
    quiet = [s for s in night_starts if not np.nansum(raw_quiet[s:s + 72])]
    # Drift is not scored at rest: successive quiet nights are ~24 h apart, its null 1 h.
    night_rows = score_windows(x, quiet, 72, groups, rng, drift=False) if quiet else []
    rest = {"nights": len(night_starts), "quiet_nights": len(quiet)}
    for k in FEATURES:
        zs = [r[k]["z"] for r in night_rows if k in r and np.isfinite(r[k]["z"])]
        if zs:
            rest[k] = {"quiet_nights": len(zs), "min_abs_z": float(np.min(np.abs(zs))),
                       "share_within_1sd": float(np.mean(np.abs(zs) <= 1.0))}
    day_starts = list(range(0, len(x) - WINDOW + 1, WINDOW))
    return {
        "label": label,
        "rows": len(x),
        "channels": len(names),
        "channels_by_group": {g: sum(1 for n in names if group[n] == g) for g in ("body", "mind", "world")},
        "windows": len(rows),
        "null_draws": N_NULL,
        "null_min_gap_buckets": int(min(r["null_gap"] for r in rows)) if rows else None,
        "features": summarize(rows),
        "rest": rest,
        "spearman": spearman_table(rows),
        "stability": daily_stability(x, day_starts),
        "leading_modes": {f"rank{i + 1}": mode_names(v, names)
                          for i, v in enumerate(wbs.top_modes(wbs.correlation(x), TOP_MODES)[1].T)},
        "seconds": round(time.time() - t0, 1),
        "_starts": starts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--postgres-uri", default=DEFAULT_POSTGRES_URI)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3290)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-layer1", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).replace(minute=0, second=0, microsecond=0)

    raw, names, group, idx, rows_seen = load(args.postgres_uri, since)
    res = wbs.tod_residuals(raw, idx)
    quiet_cols = [names.index(n) for n in ("world:juniper_messages", "world:hub_attention_beats") if n in names]

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "since": since.isoformat(),
                    "source_rows": rows_seen, "seed": args.seed}

    # Whole body: the span where bus residuals exist.
    bus_cols = [j for j, n in enumerate(names) if n.startswith("bus:")]
    bus_rows = np.flatnonzero(np.isfinite(res[:, bus_cols]).any(axis=1))
    lo, hi = int(bus_rows.min()), int(bus_rows.max()) + 1
    sel = [int(j) for j in wbs.select_channels(res[lo:hi]) if _active(raw[lo:hi, j], names[j])]
    x_all, names_all, group, merge_info = _merge(res[lo:hi][:, sel], [names[j] for j in sel], group)
    idx_all = idx[lo:hi]
    if len(quiet_cols) != 2:
        raise SystemExit("quiet-night test needs world:juniper_messages and world:hub_attention_beats")
    whole = run_block("whole_body", x_all, idx_all, names_all, group, rng, raw[lo:hi][:, quiet_cols].sum(axis=1))
    whole["span"] = [datetime.fromtimestamp(int(idx_all[0]) * BUCKET, timezone.utc).isoformat(),
                     datetime.fromtimestamp(int(idx_all[-1]) * BUCKET, timezone.utc).isoformat()]

    # Organs for #3296 and a first field-demon read.
    rank_beats = rank_shares_vs_null(x_all, whole["_starts"][::6], rng)
    scores, organ_info, stable = organs(x_all, names_all, whole["stability"]["median_similarity_by_rank"])
    demon_read: dict = {"rank_beats_null_share": rank_beats, "organs_stable": stable,
                        "organs" if stable else "preview_modes_unstable": organ_info}
    if scores.shape[1] >= 2:
        # Spec ceiling rule: if the horizon reaches K, K doubles (60 min -> 2 h).
        K = demon.DEFAULT_HORIZON
        errs = demon.rolling_errors(scores, horizon=K)
        est = demon.bootstrap_horizon(errs, n_boot=500, rng=rng)
        if est.steps >= K:
            K *= 2
            errs = demon.rolling_errors(scores, horizon=K)
            est = demon.bootstrap_horizon(errs, n_boot=500, rng=rng)
        demon_read["horizon_limit_min"] = K * 5
        cost = demon.cut_cost_vs_null(scores, demon.full_cut(scores.shape[1]), n_null=20, rng=rng, horizon=K)
        demon_read.update({
            "horizon_min": est.minutes, "horizon_p10_min": est.p10_steps * 5, "horizon_p90_min": est.p90_steps * 5,
            "skill_vs_best_baseline": [round(float(v), 4) for v in est.skill],
            "p_skill_positive": [round(float(v), 3) for v in est.p_positive],
            "full_cut_cost": [round(float(v), 4) for v in cost.cost],
            "full_cut_excess_over_null": [round(float(v), 4) for v in cost.excess],
            "full_cut_null_sd": [round(float(v), 4) for v in cost.null_sd],
            "full_cut_p_real": [round(float(v), 3) for v in cost.p_real],
            "horizon_cost_min": cost.horizon_cost_minutes,
            "refits_skipped": len(errs.skipped),
        })
    whole["organs_and_demon"] = demon_read
    whole["merged_copies"] = merge_info
    whole["channel_rule"] = ("count channels active (nonzero) in >= 90% of buckets; levels present in >= 90%; "
                             f"near-exact copies (|r| >= {wbs.COPY_THRESHOLD}) merged")
    whole.pop("_starts")
    if not args.skip_layer1:
        whole["layer1"] = layer1_top50(x_all, idx_all, names_all, group)
    report["whole_body"] = whole

    # Body only, the full --days window (the spec's 30 days of body data).
    body_cols = [j for j, n in enumerate(names) if group[n] == "body"]
    b_rows = np.flatnonzero(np.isfinite(res[:, body_cols]).any(axis=1))
    blo, bhi = int(b_rows.min()), int(b_rows.max()) + 1
    bsel = [body_cols[j] for j in wbs.select_channels(res[blo:bhi][:, body_cols])]
    xb, nb, group, body_merge = _merge(res[blo:bhi][:, bsel], [names[j] for j in bsel], group)
    body = run_block("body_only", xb, idx[blo:bhi], nb, group, rng, raw[blo:bhi][:, quiet_cols].sum(axis=1))
    body["merged_copies"] = body_merge
    body.pop("_starts")
    report["body_only"] = body

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({k: report[k]["features"] for k in ("whole_body", "body_only")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
