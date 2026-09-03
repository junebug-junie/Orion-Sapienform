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


def fetch_action_warrant(
    conn, since: datetime, until: datetime, *, limit: int | None = None
) -> TimeSeries:
    """substrate_proposal_frames.proposal_frame_json->>'action_warrant', ~2.1s cadence.

    `limit` (default None -- every existing caller, unaffected): when set,
    fetches only the `limit` MOST RECENT rows (ORDER BY ... DESC LIMIT) then
    reverses back to this function's normal ascending-time contract before
    returning -- for resolve_live_enrichment()'s live per-tick use, which
    only ever wants the single latest value (review finding, 2026-09-03: an
    unlimited ~15-minute-window query re-run every ~2s tick, at this
    channel's own ~2.1s cadence, fetches ~428 rows just to use the newest
    one). Every field here is single-valued per row (unlike
    fetch_attention_self_model(), which needs a real lookback window because
    its two field-groups can land on different rows -- limit is not offered
    there for that reason), so limit=1 is always correct for "latest",
    not just an approximation."""
    query = """
        SELECT generated_at, (proposal_frame_json ->> 'action_warrant')::float8
        FROM substrate_proposal_frames
        WHERE generated_at >= %s AND generated_at <= %s
          AND (proposal_frame_json ->> 'action_warrant') IS NOT NULL
        ORDER BY generated_at {order}
        """
    if limit is None:
        rows = _fetch(conn, query.format(order="ASC"), (since, until))
    else:
        rows = _fetch(conn, query.format(order="DESC") + " LIMIT %s", (since, until, limit))
        rows = list(reversed(rows))
    return [(ts, {"action_warrant": v}) for ts, v in rows if v is not None]


def fetch_attention_self_model(conn, since: datetime, until: datetime) -> dict[str, TimeSeries]:
    """substrate_attention_self_model.self_model_json -- heartbeat_mean_ratio (relayed from
    orion-heartbeat's /h1 over HTTP, no Postgres table of its own) + the per-domain
    prediction_error_by_domain breakdown, ~30.6s cadence.

    Returns one INDEPENDENT series per field (dict keyed by channel name, matching
    fetch_codebase_delta()'s shape), NOT one combined series. A DB row may carry only one of
    heartbeat_mean_ratio/prediction_error_by_domain (confirmed live) -- if both were merged
    into a single series entry per row (an earlier version of this function did exactly
    that), asof_forward_fill()'s single-nearest-entry lookup would silently drop whichever
    field the NEAREST row happened to lack, even when an earlier row had a real, still-valid
    value for it that correct last-observation-carried-forward should have kept. Returning
    independent per-field series and asof-joining each on its own (see fetch_all_series())
    gives every field its own correct LOCF regardless of what else that row carried."""
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
    out: dict[str, TimeSeries] = {"heartbeat_mean_ratio": []}
    for domain in _PREDICTION_ERROR_DOMAINS:
        out[f"prediction_error_{domain}"] = []
    for ts, heartbeat_raw, domains_raw in rows:
        if heartbeat_raw is not None:
            out["heartbeat_mean_ratio"].append((ts, {"heartbeat_mean_ratio": float(heartbeat_raw)}))
        if domains_raw:
            for domain in _PREDICTION_ERROR_DOMAINS:
                v = domains_raw.get(domain)
                if v is not None:
                    out[f"prediction_error_{domain}"].append((ts, {f"prediction_error_{domain}": float(v)}))
    return out


def _fetch_scalar_series(
    conn,
    *,
    table: str,
    ts_column: str,
    columns: list[tuple[str, str]],
    since: datetime,
    until: datetime,
    require_not_null: bool = False,
) -> TimeSeries:
    """Generic single-table scalar-column fetch, shared by fetch_swear_frequency,
    fetch_doc_semantic_drift, and fetch_dev_economics -- their query bodies (SELECT
    ts_column + N scalar columns, WHERE ts_column BETWEEN since/until, ORDER BY ts_column
    ASC) were near-identical copy-pasted code, differing only in table/column/channel-name
    literals (found by code review, 2026-09-02).

    `columns` is a list of (sql_column, channel_name) pairs. A row contributes a series
    entry only if at least one of its requested columns is non-NULL; each column is included
    independently (a row with column A present and column B NULL still contributes A's
    value) -- never fabricating a 0.0 for "no reading this column, this row".

    `require_not_null` (default False), when True, pushes "at least one requested column is
    non-NULL" down into the SQL WHERE clause instead of only filtering in Python -- for a
    single-column caller (fetch_swear_frequency, fetch_doc_semantic_drift) this avoids
    fetching rows that get discarded anyway (their originals, before this helper existed,
    filtered server-side too; found by code review that the first version of this shared
    helper silently dropped that pushdown, quadrupling the row count fetched over the wire
    for a signal that's ~78% NULL). Leave False for a multi-column caller where a row can be
    meaningful with only SOME columns present (fetch_dev_economics: session_count is always
    present, total_estimated_cost_usd is legitimately sometimes NULL -- neither should gate
    the other out at the SQL level)."""
    col_names = ", ".join(col for col, _channel in columns)
    where_extra = ""
    if require_not_null:
        not_null_clauses = " OR ".join(f"{col} IS NOT NULL" for col, _channel in columns)
        where_extra = f" AND ({not_null_clauses})"
    rows = _fetch(
        conn,
        f"""
        SELECT {ts_column}, {col_names}
        FROM {table}
        WHERE {ts_column} >= %s AND {ts_column} <= %s{where_extra}
        ORDER BY {ts_column} ASC
        """,
        (since, until),
    )
    out: TimeSeries = []
    for row in rows:
        ts = row[0]
        values: dict[str, float] = {}
        for (_sql_col, channel_name), value in zip(columns, row[1:]):
            if value is not None:
                values[channel_name] = float(value)
        if values:
            out.append((ts, values))
    return out


def fetch_swear_frequency(conn, since: datetime, until: datetime) -> TimeSeries:
    """juniper_affective_state_log.swear_frequency -- ~15.8min poll windows, ~6.4% nonzero
    live (2026-09-02); NULL windows (78% of them, per the model's own docstring) are dropped
    here rather than forward-filled as 0.0, since NULL means 'no message activity that window',
    not 'zero swearing' -- the two are different facts."""
    return _fetch_scalar_series(
        conn,
        table="juniper_affective_state_log",
        ts_column="observed_at",
        columns=[("swear_frequency", "swear_frequency")],
        since=since,
        until=until,
        require_not_null=True,
    )


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
    see orion/mood_arc/README.md's v4 section.

    Null-handling note: session_count is NOT NULL in the live schema (confirmed via `\\d
    dev_economics_ledger_log`), so it should never actually be NULL here -- but
    `_fetch_scalar_series()`'s generic per-column "include if non-NULL, drop if NULL"
    handling means a row with a schema-violating NULL session_count would be silently
    dropped as an ordinary missing-value row rather than surfaced as the real anomaly it
    would be (found by code review: an earlier fetch_dev_economics() unconditionally did
    `float(session_count)` with no null guard at all, so it would have crashed loudly on
    that same anomaly instead). Not re-engineered here since this function is currently
    deferred, not called by fetch_all_series() -- worth reconsidering if/when it's wired in."""
    return _fetch_scalar_series(
        conn,
        table="dev_economics_ledger_log",
        ts_column="observed_at",
        columns=[
            ("session_count", "dev_economics_session_count"),
            ("total_estimated_cost_usd", "dev_economics_total_estimated_cost_usd"),
        ],
        since=since,
        until=until,
    )


def fetch_doc_semantic_drift(conn, since: datetime, until: datetime) -> TimeSeries:
    """doc_semantic_drift_log.diff_scoped_embedding_diff, ~3.9h cadence, the sparsest of all
    13 -- rows where the hunk was unscoreable (diff_scoped_embedding_diff IS NULL, e.g.
    whitespace-only) are dropped rather than forward-filled as 0.0."""
    return _fetch_scalar_series(
        conn,
        table="doc_semantic_drift_log",
        ts_column="observed_at",
        columns=[("diff_scoped_embedding_diff", "doc_semantic_drift")],
        since=since,
        until=until,
        require_not_null=True,
    )


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
    the actual channel names trained on live inside each series' value dicts. Note
    attention_self_model contributes 5 SEPARATE entries here (one per field:
    heartbeat_mean_ratio + 4 prediction_error_{domain} channels), not one combined
    "attention_self_model" entry -- see fetch_attention_self_model()'s docstring for why."""
    return _fetch_action_warrant_and_attention_series(conn, since, until)


def _fetch_action_warrant_and_attention_series(
    conn, since: datetime, until: datetime, *, action_warrant_limit: int | None = None
) -> dict[str, TimeSeries]:
    """Shared channel-set composition behind both fetch_all_series() (offline
    batch, unlimited) and resolve_live_enrichment() (live per-tick,
    action_warrant_limit=1) -- extracted (review finding, 2026-09-03) so the
    two can't silently drift onto different channel sets if this list is
    ever revisited (fetch_all_series()'s own docstring already flags that as
    live, anticipated future work: re-including one of the currently-
    deferred sparse signals)."""
    series: dict[str, TimeSeries] = {
        "action_warrant": fetch_action_warrant(conn, since, until, limit=action_warrant_limit)
    }
    series.update(fetch_attention_self_model(conn, since, until))
    return series


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


def resolve_live_enrichment(
    conn, *, now: datetime, lookback_hours: float = 0.25
) -> dict[str, float]:
    """Live counterpart to `enrich_corpus()`, added 2026-09-03 for
    `services/orion-field-digester/app/anomaly_scorer.py`'s per-tick scoring loop.
    Resolves the SINGLE latest known value (not a forward-filled per-row series) for
    each of the 6 live-enrichment channels (`action_warrant`, `heartbeat_mean_ratio`,
    `prediction_error_{execution,chat,biometrics,bus_synaptic}`) as of `now`, over a
    narrow trailing window -- these are the only phi-v2-audit channels the live
    scorer's in-process row-building path (`collect_field_channel_pressures()`) does
    not already produce; every other v4 channel (including all 5 cabinet channels)
    reaches it the same way it reaches `field_channel_corpus.v1`, with no gap.

    `lookback_hours` defaults to 15 minutes -- `action_warrant`'s real cadence is
    ~2.1s and `attention_self_model`'s is ~30.6s (both live-confirmed in this
    module's other docstrings and re-queried on essentially every tick here), so 15
    minutes is ~29x margin over the slower cadence while keeping the query itself
    cheap and index-bound. Widen only if a live producer gap longer than that is
    actually observed -- this is a reversible keyword default, not a schema/contract
    decision.

    A channel with no reading anywhere in [now - lookback_hours, now] is left
    ENTIRELY ABSENT from the returned dict -- never fabricated as 0.0, same
    convention as `asof_forward_fill()`. The caller is responsible for treating a
    fully-empty return (all 6 channels absent) as worth logging loudly -- it can mean
    the join found nothing at all (wrong table/column, DB unreachable, a real
    producer outage longer than the lookback), not that this tick is calm.

    Uses `_fetch_action_warrant_and_attention_series()` -- the shared composition
    behind this and `fetch_all_series()` -- with `action_warrant_limit=1`: this
    function only ever wants the single latest value per channel, so fetching
    action_warrant's own full ~2.1s-cadence window (up to ~428 rows at the default
    15-minute lookback) just to discard all but the last one is pure waste (review
    finding, 2026-09-03). `attention_self_model` is NOT limited the same way -- its
    two field-groups can land on different rows (see that function's own docstring),
    so it genuinely needs the real lookback window to find each field's true latest;
    its own cadence (~30.6s) keeps that window small regardless (~29 rows at the
    default lookback, not worth optimizing further).
    """
    since = now - timedelta(hours=lookback_hours)
    series_by_name = _fetch_action_warrant_and_attention_series(
        conn, since, now, action_warrant_limit=1
    )
    latest: dict[str, float] = {}
    for series in series_by_name.values():
        if not series:
            continue
        _ts, values = series[-1]  # fetch_* queries already ORDER BY ... ASC
        latest.update(values)
    return latest
