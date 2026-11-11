from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app import reachable as R

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _bursty(days: int, seed: int = 0, local_hours=(19, 20, 21)) -> list[datetime]:
    """Juniper-like: on ~1 day in 3, a burst of 2-5 messages a few minutes apart, at an evening hour."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(days):
        if rng.random() < 0.33:
            day = (T0 + timedelta(days=d)).astimezone(R.LOCAL_TZ).replace(hour=int(rng.choice(local_hours)),
                                                                          minute=int(rng.integers(0, 40)))
            t = day.astimezone(timezone.utc)
            for _ in range(int(rng.integers(2, 6))):
                out.append(t)
                t += timedelta(minutes=int(rng.integers(5, 25)))
    return out


def test_no_history_returns_prior_for_all_three():
    f = R.forecast([], T0)
    assert f["p"] == f["p_state"] == f["p_baseline"] == R.PRIOR_POS / R.PRIOR_HOURS
    assert f["state_bucket"] == len(R.STATE_EDGES_H) and f["hours_since"] is None


def test_forecast_ignores_messages_and_typing_at_or_after_origin():
    msgs = _bursty(30)
    origin = R.hour_floor(msgs[-1])
    typed = [(m - timedelta(minutes=20), 3) for m in msgs]
    a = R.forecast(msgs, origin, typed)
    later = [origin + timedelta(minutes=m) for m in (0, 5, 70)]
    b = R.forecast([m for m in msgs if m < origin] + later, origin,
                   [t for t in typed if t[0] < origin] + [(t, 9) for t in later])
    assert a == b


def _typing_leads(days: int, seed: int = 0):
    """Messages to Orion come in hours right after Juniper typed to Claude, never otherwise; she types to
    Claude on most days, and the messages follow only some of those sessions (so cold hours exist)."""
    rng = np.random.default_rng(seed)
    msgs, typed = [], []
    for d in range(days):
        for _ in range(int(rng.integers(1, 3))):
            t = T0 + timedelta(days=d, hours=int(rng.integers(14, 23)), minutes=int(rng.integers(0, 50)))
            typed.append((t, int(rng.integers(1, 6))))
            if rng.random() < 0.4:
                msgs.append(R.hour_floor(t) + timedelta(hours=1, minutes=int(rng.integers(0, 50))))
    return sorted(msgs), sorted(typed)


def test_typing_to_claude_raises_p_in_cold_hours():
    msgs, typed = _typing_leads(60)
    origin = T0 + timedelta(days=61, hours=10)
    quiet = R.forecast(msgs, origin, typed)
    busy = R.forecast(msgs, origin, typed + [(origin - timedelta(minutes=15), 4)])
    assert quiet["state_bucket"] >= R.COLD_BUCKET and busy["state_bucket"] == quiet["state_bucket"]
    assert busy["features"]["claude_1h"] and not quiet["features"]["claude_1h"]
    assert busy["p"] > 3 * quiet["p"]


def test_asleep_hours_read_lower_than_daytime():
    # messages at daytime AND evening hours, never 03-08: asleep must be learned on its own, not via
    # the evening feature (review 2026-11-10)
    msgs = sorted(_bursty(60, local_hours=(10, 13, 16)) + _bursty(60, seed=5, local_hours=(19, 21)))
    day = (T0 + timedelta(days=61)).astimezone(R.LOCAL_TZ)
    night = R.forecast(msgs, day.replace(hour=5).astimezone(timezone.utc))
    morning = R.forecast(msgs, day.replace(hour=10).astimezone(timezone.utc))
    assert night["features"]["asleep"] and not morning["features"]["asleep"]
    assert not night["features"]["evening"] and not morning["features"]["evening"]
    assert night["state_bucket"] == morning["state_bucket"]
    assert night["p"] < morning["p"]


def test_uncovered_typing_log_is_not_read_as_not_typing():
    msgs, typed = _typing_leads(60)
    origin = T0 + timedelta(days=61, hours=10)
    full = [(T0 - timedelta(days=1), origin)]
    covered = R.forecast(msgs, origin, typed, full)
    assert covered["typing_log_covered"] and covered["p"] != covered["p_state"]
    # log down for the last 6 h: no typing evidence either way -> the state model is served
    gap = [(T0 - timedelta(days=1), origin - timedelta(hours=6))]
    down = R.forecast(msgs, origin, typed, gap)
    assert not down["typing_log_covered"] and down["p"] == down["p_state"]


def test_hours_before_the_log_existed_are_left_out_of_training():
    msgs, typed = _typing_leads(60)
    origin = T0 + timedelta(days=61, hours=10)
    # a pre-log stretch full of messages with no typing: with coverage it is masked, so the
    # typing lead keeps its full weight
    early = [T0 - timedelta(days=d, hours=4) for d in range(1, 30)]
    log = [(T0, origin)]
    busy = typed + [(origin - timedelta(minutes=15), 4)]
    masked = R.forecast(sorted(early + msgs), origin, busy, log)
    unmasked = R.forecast(sorted(early + msgs), origin, busy, [(T0 - timedelta(days=40), origin)])
    assert masked["p"] > unmasked["p"]


def test_covered_seconds_merges_overlaps():
    t = T0
    cov = R._covered([(t, t + timedelta(hours=2)), (t + timedelta(hours=1), t + timedelta(hours=3)),
                      (t + timedelta(hours=5), t + timedelta(hours=6))])
    a = np.array([R._ns(t - timedelta(hours=1)), R._ns(t + timedelta(hours=2, minutes=30))])
    b = np.array([R._ns(t + timedelta(hours=7)), R._ns(t + timedelta(hours=5, minutes=30))])
    assert list(cov(a, b)) == [4 * 3600.0, 30 * 60.0 + 30 * 60.0]


def test_state_is_hours_since_last_message_and_local_hour_is_denver():
    m = datetime(2026, 11, 10, 2, 30, tzinfo=timezone.utc)   # 19:30 MST on 11-09
    f = R.forecast([m], datetime(2026, 11, 10, 4, 0, tzinfo=timezone.utc))
    assert f["hours_since"] == 1.5 and f["state_bucket"] == 1
    assert f["local_hour"] == 21


def test_mid_conversation_reads_far_higher_than_after_two_quiet_days():
    msgs = _bursty(60)
    last = msgs[-1]
    hot = R.forecast(msgs, R.hour_floor(last) + timedelta(hours=1))
    cold = R.forecast(msgs, R.hour_floor(last) + timedelta(days=3))
    assert hot["state_bucket"] == 0 and cold["state_bucket"] == 4
    assert hot["p"] > 5 * cold["p"]


def test_outcome_window_is_half_open():
    o = T0
    assert R.outcome([o], o) == (True, 1)
    assert R.outcome([o + R.WINDOW], o) == (False, 0)
    assert R.outcome([o - timedelta(seconds=1)], o) == (False, 0)


def _replay(msgs, days):
    rows = []
    for h in range(24 * 7, 24 * days):
        o = T0 + timedelta(hours=h)
        f = R.forecast(msgs, o)
        rows.append({**f, "happened": R.outcome(msgs, o)[0]})
    return rows


def test_grade_has_no_comparison_below_min_days():
    msgs = _bursty(20)
    v = R.verdict(_replay(msgs, 20))
    assert v["all"]["vs_baseline"] is None and v["all"]["n_days"] < R.MIN_DAYS
    assert v["served"] == R.MODEL


def _replay_typed(msgs, typed, days):
    rows = []
    for h in range(24 * 7, 24 * days):
        o = T0 + timedelta(hours=h)
        rows.append({**R.forecast(msgs, o, typed), "happened": R.outcome(msgs, o)[0]})
    return rows


def test_typing_lead_gives_cold_hour_skill_on_replay():
    msgs, typed = _typing_leads(50, seed=4)
    v = R.verdict(_replay_typed(msgs, typed, 50))
    # a planted, strong signal: near-certain, and the gap is several standard errors below zero
    c = v["cold"]["vs_baseline"]
    assert c["prob_better"] > 0.99 and c["gap_per_day"] < -3 * c["gap_se"], v["cold"]


def test_bursty_history_beats_base_rate_overall_on_replay():
    msgs = _bursty(70, seed=1)
    v = R.verdict(_replay(msgs, 70))
    assert v["all"]["vs_baseline"]["prob_better"] > 0.99, v["all"]
    assert v["all"]["skill_vs_baseline"] > 0
    # cold hours exist and are graded separately
    assert v["cold"]["n_hours"] > 0 and v["cold"]["n_hours"] < v["all"]["n_hours"]


def test_random_messages_are_not_better_than_base_rate():
    rng = np.random.default_rng(3)
    msgs = sorted(T0 + timedelta(minutes=int(x)) for x in rng.integers(0, 70 * 1440, 40))
    v = R.verdict(_replay(msgs, 70))
    # nothing to learn: the grade must not claim near-certainty
    assert v["all"]["vs_baseline"]["prob_better"] < 0.9, v["all"]


def test_served_is_the_lower_loss_forecast():
    msgs = _bursty(40, seed=2)
    rows = _replay(msgs, 40)
    bad = [{**r, "p": 0.9 if not r["happened"] else 0.01} for r in rows]   # confidently wrong
    v = R.verdict(bad)
    assert v["all"]["vs_baseline"]["gap_per_day"] > 0 and v["all"]["vs_baseline"]["prob_better"] < 0.01
    ll = v["all"]["log_loss"]
    want = R.STATE_MODEL if ll["state"] < ll["baseline"] else R.BASELINE
    assert v["served"] == want
    key = {R.STATE_MODEL: "p_state", R.BASELINE: "p_baseline"}[want]
    assert R.served_p(bad[0], v["served"]) == bad[0][key]
    # with the state model also broken, the baseline is served
    worse = [{**r, "p_state": r["p"]} for r in bad]
    v2 = R.verdict(worse)
    assert v2["served"] == R.BASELINE and R.served_p(worse[0], v2["served"]) == worse[0]["p_baseline"]


def test_calibration_bins_cover_every_row():
    rows = _replay(_bursty(30), 30)
    assert sum(b["n"] for b in R.calibration(rows)) == len(rows)


class _Store:
    def __init__(self, msgs):
        self.msgs, self.rows = msgs, {}

    def juniper_messages(self, start, end):
        return [m for m in self.msgs if start <= m < end]

    def juniper_claude_typed(self, start, end):
        return []

    def juniper_typing_coverage(self, start, end):
        return [(start, end)]

    def reachable_exists(self, origin):
        return any(r["origin"] == origin for r in self.rows.values())

    def insert_reachable(self, row):
        if row["forecast_id"] in self.rows:
            return None
        self.rows[row["forecast_id"]] = dict(row)
        return row["forecast_id"]

    def pending_reachable(self, before):
        return [r for r in self.rows.values() if r["settled_at"] is None and r["window_end"] <= before]

    def settle_reachable(self, fid, happened, n, now):
        r = self.rows[fid]
        if r["settled_at"] is not None:
            return 0
        r.update(happened=happened, n_in_window=n, settled_at=now)
        return 1

    def reachable_rows(self, since, settled_only):
        return [r for r in self.rows.values() if r["origin"] >= since and (r["settled_at"] or not settled_only)]


def test_tick_issues_once_per_hour_only_early_and_settles_after_grace():
    o = datetime(2026, 11, 10, 3, 0, tzinfo=timezone.utc)
    store = _Store([o - timedelta(minutes=30), o + timedelta(minutes=20)])
    rc = R.Reachable(store)
    assert rc.tick(o + timedelta(minutes=2))["issued"] == "reachable:2026-11-10T03"
    assert rc.tick(o + timedelta(minutes=5))["issued"] is None              # already issued
    assert rc.tick(o + timedelta(minutes=30))["issued"] is None             # too late in the hour
    assert rc.tick(o + timedelta(minutes=65))["settled"] == 0               # inside the grace
    p = rc.tick(o + timedelta(hours=1, minutes=11))
    assert p["settled"] == 1
    row = store.rows["reachable:2026-11-10T03"]
    assert row["happened"] is True and row["n_in_window"] == 1
    # the 03:00 forecast saw only the 02:30 message
    assert row["n_messages"] == 1 and row["state_bucket"] == 0
    live = rc.live(o + timedelta(hours=1, minutes=12))
    assert live["current"]["served_p"] > 0 and live["grade"]["all"]["vs_baseline"] is None


def test_gap_se_is_direct_not_capped_and_matches_paired_win_p():
    from scipy import stats

    from app.backtest import paired_gap, paired_win_p
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.05, 0.1, 40)
    cand = ref - 0.03 + rng.normal(0, 0.001, 40)       # |t| far above 10
    mean, se, n = paired_gap(cand, ref)
    p, _ = paired_win_p(cand, ref)
    assert abs(mean / se) > 30
    assert p == stats.t.cdf(mean / se, df=n - 1)
    g = R._gap(cand, ref)
    assert g["gap_se"] == se and g["prob_better"] > 0.999
    same = R._gap(np.full(40, 0.25), np.full(40, 0.5))  # every day exactly 0.25 better
    assert same["gap_se"] == 0 and same["prob_better"] == 1.0
