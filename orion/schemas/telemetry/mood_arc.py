"""Mood-arc corpus row schema -- Item 1 of docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, ConfigDict

from orion.schemas.telemetry.phi_encoder import CorpusStatsV1, TrainingStatsV1


class MoodArcCorpusRowV1(BaseModel):
    """One per-tick training-data row for the not-yet-built windowed felt-
    state autoencoder (roadmap item 2). REHEARSAL status -- no cognition
    consumer by design, see orion/inner_state_registry.py.

    dominant_node was removed 2026-08-21. It was null only on ticks where the
    phi encoder didn't run (disabled/degraded/failed), via
    _dominant_hardware_node() inside the former handle_self_state()'s
    encoder-success branch (Phase 2/3), and had already been unconditionally
    null on every row since 2026-07-22 (SelfStateV1 burn removed its only
    input, dominant_attention_target_details, with no FieldAttentionFrameV1-
    native replacement built). orion-spark-introspector -- the field's sole
    producer -- was retired outright 2026-07-28, so the column is dropped
    rather than left as a permanent null.

    Rotation (2026-07-13): the sink this writes through
    (InnerStateCorpusSink, shared with INNER_FEATURES_CORPUS_PATH) now
    rotates at CORPUS_SINK_MAX_BYTES (default 200MB) and keeps at most
    CORPUS_SINK_ROTATED_KEEP (default 5) rotated siblings -- see
    orion/telemetry/corpus_sink.py (promoted here 2026-07-13 from
    services/orion-spark-introspector/app/inner_state_sink.py when
    orion-field-digester's field_channel_corpus.v1 became a second
    consumer). Unlike
    InnerStateFeaturesV1 (recoverable via scripts/backfill_phi_corpus.py
    from Postgres), there is NO backfill path for pruned mood-arc rows --
    once a rotated file ages past the retention count, that slice of
    history is genuinely gone, not just archived. At the default policy
    (200MB x 5 = up to ~1GB retained) this is generous relative to the
    "weeks, not months" scope roadmap item 2 needs, but is a real,
    permanent-not-recoverable loss if collection runs far longer than
    that unattended.
    """

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    self_state_id: str
    coherence: float
    energy: float
    novelty: float
    valence: float
    valence_source: Literal["proxy", "heuristic"]


class MoodArcEncoderManifestV1(BaseModel):
    """Item 2's windowed felt-state-trajectory encoder manifest -- dark
    artifact, disk-only, no bus publish of its own (see roadmap item 2,
    docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md).

    **Not consumer-free, corrected 2026-09-02**: the trained artifact this
    manifest describes is loaded and scored by
    services/orion-field-digester/app/anomaly_scorer.py (imports
    orion.mood_arc.fit_encoder directly), which feeds the Hub's main-page
    Cognitive EKG viz -- gated behind FIELD_CHANNEL_ANOMALY_ENABLED (default
    off). **Converged 2026-09-03**: that consumer now resolves this same
    models_root's active.json directly (FIELD_CHANNEL_ANOMALY_MODELS_ROOT,
    renamed from FIELD_CHANNEL_ANOMALY_ENCODER_DIR) rather than a separately-
    tracked directory -- whatever this module's own promote() subcommand
    last activated is what the live consumer reads on its next restart, plus
    live per-tick enrichment for the channels its in-process row-building
    alone can't produce. See orion/mood_arc/README.md's
    "Status" note for the full chain and both caveats.

    2026-07-13 methodology addition (orion/mood_arc/fit_encoder.py, same
    session as this manifest's initial fields): the spec's original single
    shuffle-gate design was found to be too weak on its own -- the corpus's
    real autocorrelation is largely explained by a known, deliberate
    leaky-integrator decay mechanism (BIOMETRICS_FIELD_DECAY_RATE=0.92 in
    services/orion-field-digester/app/digestion/decay.py), so an encoder can
    pass the shuffle floor purely by learning that already-known mechanism
    without capturing anything specific to Orion's actual trajectories. The
    fields below extend the manifest with a second, non-gating "ceiling"
    comparison against a matched-autocorrelation AR(1) surrogate, plus a
    purged/embargoed temporal train/held-out split (naive random window
    sampling leaks given ~10-15 tick autocorrelation from 50%-overlapping
    windows) and a block-bootstrap confidence interval on the floor ratio.
    None of this is in the original written spec doc -- it is stricter than
    what item 2 originally asked for, added after empirical spike work found
    the original single-gate design passed for the wrong reason.

    2026-07-17 corpus-swap rework (orion/mood_arc/fit_encoder.py, same
    session): this script now trains against field_channel_corpus.v1
    (orion.schemas.telemetry.field_channel_corpus.FieldChannelCorpusRowV1)
    instead of mood_arc_corpus.v1 -- a variable-width channel_name -> value
    dict rather than a fixed 4-field schema, so the field set a given run
    actually trained over is no longer implied by a module-level constant.
    channel_names (below) makes each manifest self-describing about its own
    feature set (the output of that run's select_fields()/
    prune_correlated_fields() calls), since two runs against the same
    corpus can legitimately end up with different channel sets depending on
    --variance-eps/--corr-threshold or which channels were active that
    training window.
    """

    model_config = ConfigDict(extra="forbid")

    encoder_id: str
    encoder_version: str
    parent_version: Optional[str] = None
    status: Literal["candidate", "active", "retired"]
    architecture: str  # "mlp_shallow_v1", same as phi encoder
    window_size: int
    stride: int
    max_gap_sec: float
    hidden_dim: int
    latent_dim: int
    # Channel names (post select_fields()/prune_correlated_fields()) that
    # this run's windows/AR(1) fit/training actually used, in the fixed
    # order their values are flattened into each window vector. Required,
    # not Optional -- every manifest orion/mood_arc/fit_encoder.py writes
    # from 2026-07-17 onward populates this; there is no prior real
    # (non-scratch) manifest artifact from before this field existed to
    # stay backward-compatible with (this is a dark, disk-only artifact,
    # never committed to the repo).
    channel_names: list[str]
    corpus: CorpusStatsV1        # reused as-is from orion.schemas.telemetry.phi_encoder
    training: TrainingStatsV1    # reused as-is
    shuffle_baseline_loss: float # held_out_loss with rows shuffled within-window (see gate)
    # floor_ratio / floor_pass (2026-07-27, previously computed by cmd_train's
    # two_tier_gate() and only ever printed to stdout, never persisted --
    # found while building cmd_promote, which needs a durable, reloadable
    # answer to "did this exact candidate pass its own gate" rather than
    # trusting an operator's memory of a training run's console output).
    # floor_ratio = real_held_loss / shuffle_baseline_loss; floor_pass =
    # floor_ratio < FLOOR_MAX_RATIO, the one hard gate this project has
    # (ceiling_ratio, above, is diagnostic only). None only for manifests
    # written before this field existed.
    floor_ratio: Optional[float] = None
    floor_pass: Optional[bool] = None
    # purge_gap_windows: number of windows dropped as an embargo zone between
    # the train/held-out temporal boundary (see purged_temporal_split()) --
    # not in the original spec, added because a held-out window merely
    # adjacent to a training window is still autocorrelation-leaked even
    # with zero literal tick overlap (measured ACF stays nonzero out to lag
    # ~10-15 ticks, ~20-30s). Optional/None only for manifests written before
    # this methodology addition -- orion/mood_arc/fit_encoder.py always
    # populates a real value; None is never fabricated as 0 (0 would falsely
    # claim "no purge zone was used", a real and different config choice).
    purge_gap_windows: Optional[int] = None
    # held_out_blocks: number of time-distributed held-out blocks used by
    # block_purged_temporal_split() (fit_encoder.py). 1 (or None, for
    # manifests written before this field existed) means the original
    # single-trailing-block methodology (v3's validated behavior).
    # Confirmed live 2026-09-02: a single trailing block on a wide
    # (multi-day) window can be drawn from a materially different operating
    # regime than train (prediction_error's mean shifted +1.73 standard
    # deviations between a 3.4-day corpus's first-85%/last-15% slices),
    # inflating floor_ratio/ceiling_ratio for reasons unrelated to learned
    # structure. >1 spreads held-out across that many equal time segments
    # instead. Optional/None only for manifests written before this
    # methodology addition; fit_encoder.py always populates a real value
    # (>=1) once written -- None is never fabricated as 1.
    held_out_blocks: Optional[int] = None
    # ar1_surrogate_loss: held-out reconstruction loss against synthetic
    # windows generated from a per-channel AR(1) null model fit on the
    # training portion only (see generate_ar1_surrogate_windows()) -- the
    # "this is already explained by the known decay filter" null. None means
    # not computed for this manifest, never a fabricated 0.0.
    ar1_surrogate_loss: Optional[float] = None
    # ceiling_ratio: real_held_loss / ar1_surrogate_loss. Diagnostic and
    # exploratory ONLY, not a hard gate -- there is no calibrated pass/fail
    # threshold for this yet across multiple training runs. Recorded here so
    # future runs can be compared once enough runs exist to calibrate one.
    # Do not read a low/high ceiling_ratio as pass/fail; only floor_ratio's
    # derived floor_pass (see two_tier_gate()) is a hard gate today. None
    # means not computed for this manifest, never a fabricated 0.0.
    ceiling_ratio: Optional[float] = None
    # floor_ratio_ci_low / floor_ratio_ci_high: 95% block-bootstrap
    # confidence interval on real_held_loss / shuffle_baseline_loss
    # (block_bootstrap_ratio_ci()), resampling contiguous blocks of
    # held-out windows rather than individual windows (they're
    # autocorrelated, so naive i.i.d. bootstrap would overstate confidence).
    # None means not computed for this manifest, never a fabricated 0.0.
    floor_ratio_ci_low: Optional[float] = None
    floor_ratio_ci_high: Optional[float] = None
    git_sha: str
    trained_at: datetime
    promoted_at: Optional[datetime] = None
