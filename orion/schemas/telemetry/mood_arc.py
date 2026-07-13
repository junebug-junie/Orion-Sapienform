"""Mood-arc corpus row schema -- Item 1 of docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, ConfigDict

from orion.schemas.telemetry.phi_encoder import CorpusStatsV1, TrainingStatsV1


class MoodArcCorpusRowV1(BaseModel):
    """One per-tick training-data row for the not-yet-built windowed felt-
    state autoencoder (roadmap item 2). REHEARSAL status -- no cognition
    consumer by design, see orion/self_state/inner_state_registry.py.

    dominant_node is null on any tick where the phi encoder didn't run
    (disabled/degraded/failed) -- _dominant_hardware_node() is only called
    inside handle_self_state()'s encoder-success branch, an existing
    characteristic inherited from Phase 2/3, not introduced here. Do not
    read a null dominant_node here as "no salient node this tick"; it may
    just mean the encoder was down.

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
    dominant_node: Optional[str] = None


class MoodArcEncoderManifestV1(BaseModel):
    """Item 2's windowed felt-state-trajectory encoder manifest -- dark
    artifact, disk-only, no cognition consumer yet (see roadmap item 2,
    docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md).

    2026-07-13 methodology addition (scripts/fit_mood_arc_encoder.py, same
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
    corpus: CorpusStatsV1        # reused as-is from orion.schemas.telemetry.phi_encoder
    training: TrainingStatsV1    # reused as-is
    shuffle_baseline_loss: float # held_out_loss with rows shuffled within-window (see gate)
    # purge_gap_windows: number of windows dropped as an embargo zone between
    # the train/held-out temporal boundary (see purged_temporal_split()) --
    # not in the original spec, added because a held-out window merely
    # adjacent to a training window is still autocorrelation-leaked even
    # with zero literal tick overlap (measured ACF stays nonzero out to lag
    # ~10-15 ticks, ~20-30s). Optional/None only for manifests written before
    # this methodology addition -- scripts/fit_mood_arc_encoder.py always
    # populates a real value; None is never fabricated as 0 (0 would falsely
    # claim "no purge zone was used", a real and different config choice).
    purge_gap_windows: Optional[int] = None
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
