"""Corpus enrichment for the mood-arc v4 retrain.

Joins signals identified by `docs/superpowers/specs/2026-08-21-phi-v2-design.md`'s
live-metric audit that do NOT already reach `field_channel_corpus.v1` through
`orion.field.pressure.collect_field_channel_pressures()` (unlike the field-digester
`node_vectors`/`capability_vectors` channels -- including the cabinet sensors and the mic,
`cabinet_ambient_audio_activity` -- which are already present; see `orion/mood_arc/README.md`'s
v4 section) onto an existing `field_channel_corpus.v1` JSONL corpus, by
last-observation-carried-forward ("asof") join on each signal's own real *occurrence-time*
column -- never `created_at`. Every source table here has both: `created_at` is insert/write
time, `generated_at`/`observed_at` is when the thing actually happened (confirmed live,
2026-09-02, via `\\d <table>` against `conjourney`). Joining on `created_at` would silently
shift every signal by its own write-latency, which varies per producer.

Of the ~13 signals the audit named, `fetch_all_series()` (below) only wires in the ones dense
enough to carry real per-window trajectory information at `fit_encoder.py`'s default
`window_size`/cadence -- `action_warrant` and `attention_self_model` (heartbeat_mean_ratio +
per-domain prediction-error). The sparser ones (git/pr/graph_delta, dev_economics,
doc_semantic_drift, swear_frequency) are fully implemented and tested here but deliberately
NOT called by `fetch_all_series()` -- see that function's docstring for why (cadence far
coarser than the window, empirically confirmed to fail the floor gate when included naively).
Metric-shape note, not fixed here: `prediction_error_*`/`git/pr/graph_delta` and the mic
channel (`cabinet_ambient_audio_activity`, in `orion/telemetry/ambient_audio.py`) are all
baseline-relative deviation/z-score measures (bounded, but measuring change from an adaptive
recent baseline, not absolute level) -- a structurally different family from mood_arc's
existing absolute-level-with-decay channels (`cpu_pressure` etc.), which the AR(1) ceiling
gate's methodology was built around. A sustained-but-unchanging state reads as "calm" on a
deviation-family channel by construction (hand-verified for the mic:
`orion/telemetry/ambient_audio.py`'s own docstring proves `activity == 0.0` for an exactly
constant raw level, however loud). Worth naming plainly in any read of what a "calm" v4
reading actually means for those specific channels.

A corpus tick strictly before a signal's first known value in the queried window is left with
that channel key entirely ABSENT here -- `fit_encoder.py`'s own `build_windows()`/
`select_fields()`/`_channel_stat_matrix()` already fill genuine absence with `0.0` downstream.
Writing an explicit `0.0` in this module instead would make "no reading yet" indistinguishable
from "read a real zero" -- exactly the bug class this repo's own metric-quality history warns
about (an `x or DEFAULT` silently eating a configured zero; a decayed-to-zero artifact reading
identical to genuinely-calm-at-zero). `asof_forward_fill()` reports a per-series missing count
instead so a join that silently failed end-to-end (wrong table/column/cadence) is visibly
different from one that's just early or genuinely sparse.

Dark/offline tooling only -- no bus publish. This module only ever reads Postgres (via the
canonical `orion.db_readonly.open_readonly_connection`, read-only-enforced) and returns data
for `fit_encoder.py`'s `enrich-corpus` subcommand to write out as a plain JSONL file.
"""
from __future__ import annotations

import bisect
from datetime import datetime, timedelta

from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1

# One (timestamp, {channel_name: value}) entry per real reading, sorted ascending by timestamp.
TimeSeries = list[tuple[datetime, dict[str, float]]]

_PREDICTION_ERROR_DOMAINS = ("execution", "chat", "biometrics", "bus_synaptic")

# substrate_codebase_delta_log.domain -> the channel name we train on. Kept
# distinct from the raw domain string so `git_delta`/`pr_lifecycle_delta`/
# `graph_delta` read as channel names in a manifest, matching this repo's
# other `*_delta`/`*_pressure` naming.
_CODEBASE_DELTA_CHANNELS = {
    "git": "git_delta",
    "pr_lifecycle": "pr_lifecycle_delta",
    "graph": "graph_delta",
}


def _fetch(conn, query: str, params: tuple) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def fetch_action_warrant(conn, since: datetime, until: datetime) -> TimeSeries:
    """substrate_proposal_frames.proposal_frame_json->>'action_warrant', ~2.1s cadence."""
    rows = _fetch(
        conn,
        """
        SELECT generated_at, (proposal_frame_json ->> 'action_warrant')::float8
        FROM substrate_proposal_frames
        WHERE generated_at >= %s AND generated_at <= %s
          AND (proposal_frame_json ->> 'action_warrant') IS NOT NULL
        ORDER BY generated_at ASC
        """,
        (since, until),
    )
    return [(ts, {"action_warrant": v}) for ts, v in rows if v is not None]


def fetch_attention_self_model(conn, since: datetime, until: datetime) -> TimeSeries:
    """substrate_attention_self_model.self_model_json -- heartbeat_mean_ratio (relayed from
    orion-heartbeat's /h1 over HTTP, no Postgres table of its own) + the per-domain
    prediction_error_by_domain breakdown, ~30.6s cadence. A row may carry only one of the two
    groups; both are merged into the same series entry when present."""
    rows = _fetch(
        conn,
        """
        SELECT generated_at, self_model_json ->> 'heartbeat_mean_ratio',
               self_model_json -> 'prediction_error_by_domain'
        FROM substrate_attention_self_model
        WHERE generated_at >= %s AND generated_at <= %s
        ORDER BY generated_at ASC
        """,
        (since, until),
    )
    out: TimeSeries = []
    for ts, heartbeat_raw, domains_raw in rows:
        values: dict[str, float] = {}
        if heartbeat_raw is not None:
            values["heartbeat_mean_ratio"] = float(heartbeat_raw)
        if domains_raw:
            for domain in _PREDICTION_ERROR_DOMAINS:
                v = domains_raw.get(domain)
                if v is not None:
                    values[f"prediction_error_{domain}"] = float(v)
        if values:
            out.append((ts, values))
    return out


def fetch_swear_frequency(conn, since: datetime, until: datetime) -> TimeSeries:
    """juniper_affective_state_log.swear_frequency -- ~15.8min poll windows, ~6.4% nonzero
    live (2026-09-02); NULL windows (78% of them, per the model's own docstring) are dropped
    here rather than forward-filled as 0.0, since NULL means 'no message activity that window',
    not 'zero swearing' -- the two are different facts."""
    rows = _fetch(
        conn,
        """
        SELECT observed_at, swear_frequency
        FROM juniper_affective_state_log
        WHERE observed_at >= %s AND observed_at <= %s
          AND swear_frequency IS NOT NULL
        ORDER BY observed_at ASC
        """,
        (since, until),
    )
    return [(ts, {"swear_frequency": float(v)}) for ts, v in rows]


def fetch_codebase_delta(conn, since: datetime, until: datetime) -> dict[str, TimeSeries]:
    """One independent series per substrate_codebase_delta_log domain. Each domain updates on
    its own real cadence (git ~6.6h, pr_lifecycle ~22min, graph ~11.6h as of the 2026-09-02
    live audit) -- merging them into one series would let a stale domain's forward-filled
    value get silently overwritten by an unrelated domain's update, which is not what
    'domain X's score as of now' means. Returns {channel_name: series}, one entry per
    _CODEBASE_DELTA_CHANNELS value, so callers can asof-join each independently."""
    out: dict[str, TimeSeries] = {name: [] for name in _CODEBASE_DELTA_CHANNELS.values()}
    rows = _fetch(
        conn,
        """
        SELECT domain, observed_at, score
        FROM substrate_codebase_delta_log
        WHERE observed_at >= %s AND observed_at <= %s AND domain = ANY(%s)
        ORDER BY observed_at ASC
        """,
        (since, until, list(_CODEBASE_DELTA_CHANNELS.keys())),
    )
    for domain, ts, score in rows:
        channel = _CODEBASE_DELTA_CHANNELS.get(domain)
        if channel is not None and score is not None:
            out[channel].append((ts, {channel: float(score)}))
    return out


def fetch_dev_economics(conn, since: datetime, until: datetime) -> TimeSeries:
    """dev_economics_ledger_log -- session_count/total_estimated_cost_usd, ~16.4min cadence.
    total_estimated_cost_usd is nullable (unpriced sessions).

    `total_tokens` is deliberately NOT included here. Live-checked (2026-09-02, the window
    this v4 retrain actually trains on): it ranges 0-59,290,459 (avg ~4.07M) -- every other
    channel in this corpus is roughly a [0,1]-scaled pressure/ratio (a first v4 training run
    that DID include raw total_tokens confirmed the effect empirically: channel_variance for
    it came back at ~9.2e13 against every other channel's 1e-6-1.3 range, and the resulting
    floor/ceiling gate numbers moved in a way consistent with the shared MLP's shared-scale
    MSE reconstruction loss being dominated by reconstructing that one channel's raw
    magnitude, not real trajectory structure across the pressure vector). This is a real,
    load-bearing exclusion, not a values judgment about the signal's worth -- session_count
    (0-7) and total_estimated_cost_usd ($0.07-$33) stayed in because their live ranges are
    within a defensible order of magnitude of the rest of the corpus; total_tokens is not.
    Properly normalizing it (rather than dropping it) is real follow-up work, not done here --
    see orion/mood_arc/README.md's v4 section."""
    rows = _fetch(
        conn,
        """
        SELECT observed_at, session_count, total_estimated_cost_usd
        FROM dev_economics_ledger_log
        WHERE observed_at >= %s AND observed_at <= %s
        ORDER BY observed_at ASC
        """,
        (since, until),
    )
    out: TimeSeries = []
    for ts, session_count, cost in rows:
        values: dict[str, float] = {"dev_economics_session_count": float(session_count)}
        if cost is not None:
            values["dev_economics_total_estimated_cost_usd"] = float(cost)
        out.append((ts, values))
    return out


def fetch_doc_semantic_drift(conn, since: datetime, until: datetime) -> TimeSeries:
    """doc_semantic_drift_log.diff_scoped_embedding_diff, ~3.9h cadence, the sparsest of all
    13 -- rows where the hunk was unscoreable (diff_scoped_embedding_diff IS NULL, e.g.
    whitespace-only) are dropped rather than forward-filled as 0.0."""
    rows = _fetch(
        conn,
        """
        SELECT observed_at, diff_scoped_embedding_diff
        FROM doc_semantic_drift_log
        WHERE observed_at >= %s AND observed_at <= %s
          AND diff_scoped_embedding_diff IS NOT NULL
        ORDER BY observed_at ASC
        """,
        (since, until),
    )
    return [(ts, {"doc_semantic_drift": float(v)}) for ts, v in rows]


def fetch_all_series(conn, since: datetime, until: datetime) -> dict[str, TimeSeries]:
    """One named series per v4 input group actually wired into training. Deliberately
    NARROWER than the full phi-v2 audit list: only signals dense enough to carry real
    trajectory information inside `fit_encoder.py`'s window (`window_size=30` at ~2s/tick
    default, ~60s span) are included here --

      - action_warrant: ~2.1s cadence, well inside the window.
      - attention_self_model (heartbeat_mean_ratio + per-domain prediction-error): ~30.6s
        cadence -- coarser, but still 1-2 real transitions per window, not constant.

    `fetch_swear_frequency`/`fetch_codebase_delta`/`fetch_dev_economics`/
    `fetch_doc_semantic_drift` are deliberately NOT called here even though they're fully
    implemented and tested above. Their real update cadences (~16min-~11.6h, confirmed live
    2026-09-02) are far coarser than the window itself -- last-observation-carried-forward
    into a 60s window makes them constant across nearly every window, which is not real
    per-window trajectory signal, just added input dimensionality with no matching capacity
    increase to justify it. A first v4 training run that DID include them (git history, this
    module's earlier version) failed the floor gate (floor_ratio=0.68 vs the <0.5 threshold,
    ceiling_ratio=1.31 -- worse than a trivial AR(1) surrogate) after an unrelated scale bug
    was already fixed, consistent with this being a real capacity/structure problem, not
    noise. Wiring these back in is real follow-up work needing a DIFFERENT representation --
    e.g. one static per-window "context" value instead of a repeated-30x trajectory slot --
    not a bigger window or more epochs on the current shape. See orion/mood_arc/README.md's
    v4 section.

    The dict key is only for reporting which query/table a missing-value count belongs to --
    the actual channel names trained on live inside each series' value dicts."""
    return {
        "action_warrant": fetch_action_warrant(conn, since, until),
        "attention_self_model": fetch_attention_self_model(conn, since, until),
    }


def asof_forward_fill(rows: list[FieldChannelCorpusRowV1], series: TimeSeries) -> int:
    """Merges `series` onto `rows`' `channels` dicts in place, by
    last-observation-carried-forward on `row.generated_at`. `rows` and `series` must each
    already be sorted ascending by timestamp (`_load_jsonl` sorts rows; the `fetch_*` queries
    above `ORDER BY ... ASC`) -- re-sorting `series` defensively here since callers may combine
    series from more than one query.

    Returns the count of rows left with none of this series' keys added (because
    `row.generated_at` was before the series' first entry in the queried window). A caller
    should report this per series, not just log it -- a series where 100% of rows are missing
    means the join found nothing at all (wrong table/column/cadence, or the lookback window
    didn't reach far enough back), not that the signal is calm.
    """
    if not series:
        return len(rows)
    series = sorted(series, key=lambda entry: entry[0])
    timestamps = [ts for ts, _ in series]
    missing = 0
    for row in rows:
        idx = bisect.bisect_right(timestamps, row.generated_at) - 1
        if idx < 0:
            missing += 1
            continue
        _, values = series[idx]
        row.channels.update(values)
    return missing


def enrich_corpus(
    rows: list[FieldChannelCorpusRowV1],
    conn,
    *,
    lookback_hours: float,
) -> dict[str, int]:
    """Enriches `rows` in place (mutates each row's `channels` dict) with the phi-v2 audit's
    ~13 signals not already present via field-digester's node_vectors/capability_vectors
    merge. `lookback_hours` widens the Postgres query window before `rows`' own earliest
    timestamp, so the corpus's first ticks can still find a real prior value to forward-fill
    from rather than starting genuinely absent -- all 6 source tables here have rows well
    predating `field_channel_corpus.v1`'s current (rotated) file, so a modest lookback is
    enough in practice.

    Returns `{series_name: rows_left_without_a_value}` for the caller to report.
    """
    if not rows:
        return {}
    since = min(r.generated_at for r in rows) - timedelta(hours=lookback_hours)
    until = max(r.generated_at for r in rows)
    series_by_name = fetch_all_series(conn, since, until)
    return {name: asof_forward_fill(rows, series) for name, series in series_by_name.items()}
