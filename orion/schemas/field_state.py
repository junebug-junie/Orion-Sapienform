from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FieldEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    edge_type: Literal["node_capability", "node_service", "service_organ", "capability_cognitive", "node_dependency", "capability_capability"]
    weight: float = Field(ge=0.0, le=1.0)
    channel_map: dict[str, str] = Field(default_factory=dict)
    weight_source: Literal["designed", "learned"] = "designed"
    learned_at: datetime | None = None


class FieldStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["field.state.v1"] = "field.state.v1"
    generated_at: datetime
    tick_id: str
    node_vectors: dict[str, dict[str, float]] = Field(default_factory=dict)
    capability_vectors: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Phase 3 (2026-07-12, self-state/mesh substrate redesign): keyed by
    # capability target_id -> channel -> the edge source_id (a node_id like
    # "node:atlas" or another capability_id for a capability_capability edge)
    # that contributed the single largest weighted amount to that channel in
    # the most recent diffusion pass (services/orion-field-digester/app/
    # digestion/diffusion.py). A simple "primary contributor this tick"
    # proxy, not a full historical attribution ledger -- capability_vectors
    # values accumulate across ticks, but tracking exact per-tick provenance
    # of an accumulated value would be a bigger feature than this phase
    # needs. Lets self-state's evidence say which node is behind a pressure
    # instead of an anonymous capability-level number.
    capability_provenance: dict[str, dict[str, str]] = Field(default_factory=dict)
    edges: list[FieldEdgeV1] = Field(default_factory=list)
    recent_perturbations: list[str] = Field(default_factory=list)
    # Wall-clock timestamps parallel to recent_perturbations (same length,
    # same index alignment -- recent_perturbation_at[i] is when
    # recent_perturbations[i] was first seen). Added 2026-07-16 alongside the
    # saturating-counter fix: services/orion-field-digester/app/digestion/
    # perturbation.py now prunes recent_perturbations by wall-clock recency
    # instead of a fixed last-20-labels-ever cap (the old cap saturated
    # self_state's recent_perturbation_count to 1.0 within the first few
    # ticks of any real corpus and then pinned it there permanently -- see
    # perturbation.py's module docstring for the live evidence). Both lists
    # are always written/pruned together; a FieldStateV1 persisted before
    # this fix will load with an empty recent_perturbation_at, so the length
    # mismatch resets recent_perturbations to empty on the very first tick
    # after upgrade (zip() in perturbation.py's pruning stops at the shorter
    # list) -- an immediate, one-tick reset, not gated by window expiry --
    # and it repopulates normally from there.
    recent_perturbation_at: list[datetime] = Field(default_factory=list)
    # Baseline-relative recent_perturbation salience (2026-07-28): fixes a
    # live-confirmed saturation bug -- both self_state/scoring's old
    # count/10.0 salience and this file's neighboring recent_perturbation_count
    # channel (orion/field/pressure.py, count/20.0) turned out to be
    # permanently pinned at their 1.0 ceiling under real live traffic (steady
    # state observed 2026-07-28 was ~100-118 distinct labels in the 60s
    # window above, 5-10x past both fixed caps) -- an absolute magic-number
    # cap can't discriminate a real burst from normal mesh churn once normal
    # churn already exceeds the cap. Same EWMA-baseline-vs-z-score
    # methodology as bus_synaptic_prediction_error's gap_zscore (orion/bus/
    # ewma.py::compute_ewma_update, reused not reimplemented), applied here
    # to the recent_perturbations *count* itself rather than per-edge timing.
    # Updated once per apply_perturbations() call (services/orion-field-
    # digester/app/digestion/perturbation.py) using this tick's post-prune
    # len(recent_perturbations) as the observed value. zscore is None on the
    # very first call (recent_perturbation_ewma_n becomes 1 that same tick,
    # with no prior baseline to compare against yet) and numeric from the
    # second call onward -- same "no empty-shell cognition" rule
    # compute_ewma_update already documents for its own first-observation
    # case.
    #
    # Known accepted limitation (not fixed by
    # orion/field/pressure.py::RECENT_PERTURBATION_EWMA_MIN_SAMPLES, which
    # only guards the separate few-sample variance-noise problem): on any
    # genuinely fresh FieldStateV1 -- first-ever deploy, or a store
    # wipe/migration (not hypothetical here; see the 2026-07-23 Postgres
    # disk-death incident that required a full re-feed) -- both
    # recent_perturbations and this EWMA baseline start from empty together,
    # so the reading stays elevated for the first several minutes of real
    # wall-clock time regardless of whether anything is actually anomalous
    # (hand-simulated: still >0.6 at tick 34 of a steady 3-labels/2s-tick
    # baseline). This is a smaller, time-boxed, self-healing recurrence of
    # the same failure shape this fix otherwise kills for the steady-state
    # (i.e. normal, long-running-service) case, which is genuinely fixed and
    # independently verified. Accepted rather than engineered away here to
    # keep this patch thin -- a real follow-up would need to gate on
    # elapsed wall-clock time since state init, not just sample count.
    recent_perturbation_ewma: float = 0.0
    recent_perturbation_ewma_var: float = 0.0
    recent_perturbation_ewma_n: int = 0
    recent_perturbation_zscore: float | None = None
    # Decay/injection-interval mismatch fix (2026-07-17): keyed node_id ->
    # channel -> the wall-clock timestamp of that channel's last real write
    # from apply_perturbations() (services/orion-field-digester/app/
    # digestion/perturbation.py). Mirrors node_vectors' own node_id -> channel
    # shape, one level deeper only in value type (datetime instead of float).
    # apply_decay() (app/digestion/decay.py) uses this to hold a channel's
    # value flat while it is still fresh and only apply this tick's decay
    # factor once the channel has gone genuinely stale, instead of decaying
    # every NODE_DECAY_CHANNELS entry unconditionally every 2s tick regardless
    # of whether new data arrived. Same backward-compatibility precedent as
    # recent_perturbation_at (2026-07-16): a FieldStateV1 persisted before
    # this fix loads with an empty dict, so every existing channel is "unknown
    # freshness" until its next real perturbation -- apply_decay() treats a
    # missing entry as the safe default (decay as before today).
    node_vector_updated_at: dict[str, dict[str, datetime]] = Field(default_factory=dict)
    # Precision-weighted proposal confidence (2026-07-28, docs/superpowers/
    # specs/2026-07-28-precision-weighted-proposal-scoring-design.md):
    # per-PRESSURE_DIMENSION (orion.proposals.scoring.PRESSURE_DIMENSIONS)
    # EWMA mean/variance/z-score baseline, replacing orion/proposals/
    # scoring.py::dimension_confidence()'s old binary presence flag
    # (`1.0 if dimension_id in field_pressures else 0.0`). Same EWMA-
    # baseline-vs-z-score methodology as recent_perturbation_zscore above and
    # bus_synaptic_prediction_error's gap_zscore (orion/bus/ewma.py::
    # compute_ewma_update, reused not reimplemented). Real historical
    # variance was measured first, not assumed (scripts/analysis/measure_
    # proposal_dimension_variance.py, 2026-07-28): all four PRESSURE_
    # DIMENSIONS showed real, non-degenerate variance and real, periodically-
    # recurring perturbation-driven refreshes over a real 16h substrate_
    # field_state window (none are the node:substrate.route "zero real
    # events ever" disease) -- see that script's report for the numbers.
    #
    # Updated once per digestion tick (services/orion-field-digester/app/
    # digestion/precision.py::update_dimension_precision_baseline(), called
    # from run_digestion_tick() after decay/diffusion/suppression so it
    # scores the same final-for-this-tick orion.field.pressure.
    # field_pressures(state) reading that orion/proposals/builder.py later
    # consumes) using that reading as the observed value per dimension.
    #
    # Dict-keyed by dimension_id rather than 4 separate scalar fields (unlike
    # recent_perturbation_ewma* above) since PRESSURE_DIMENSIONS is a real,
    # documented-as-not-guaranteed-stable set (its own module comment notes
    # 3 dimensions were already removed in the 2026-07-22 SelfStateV1 burn) --
    # a dict avoids hardcoding N near-identical scalar fields and keeps this
    # schema stable if that set changes again. A dimension absent from
    # field_pressures() this tick is NOT updated that tick (its existing
    # baseline holds -- "absent this tick" is a real fact, not a 0.0 to
    # fabricate and absorb into the baseline). All 4 dicts default to `{}`
    # for backward-compat with any FieldStateV1 persisted before this field
    # existed -- dimension_confidence() treats a dimension missing from
    # `dimension_precision_ewma_n` as `n=0` (cold start), so an upgraded
    # persisted row degrades to "no confidence signal yet" rather than
    # raising or fabricating a reading.
    dimension_precision_ewma: dict[str, float] = Field(default_factory=dict)
    dimension_precision_ewma_var: dict[str, float] = Field(default_factory=dict)
    dimension_precision_ewma_n: dict[str, int] = Field(default_factory=dict)
    dimension_precision_zscore: dict[str, float] = Field(default_factory=dict)
    # Field deviation tension (2026-08-16, docs/superpowers/specs/2026-08-16-
    # tension-driven-mutating-dispatch-design.md): per-(node_id, channel) EWMA
    # baseline for orion.attention.tension.DeviationGate, carried tick-to-tick
    # exactly like dimension_precision_ewma* above (same backward-compat
    # contract -- a key absent here is cold start, not a fabricated 0.0).
    # Flat-keyed as f"{node_id}\x1f{channel}" (0x1f, ASCII unit separator:
    # real node_ids already contain colons and dots, e.g.
    # "node:substrate.route", ruling those out). Bounded cardinality (4 nodes
    # x 33 channels = 132 keys, live 2026-08-14) -- NOT unbounded per-event
    # growth like the evidence_event_ids TOAST incident CLAUDE.md records;
    # a new node or channel adds a bounded number of keys, ticks do not.
    # Written by services/orion-field-digester/app/digestion/tension.py.
    tension_baseline_mu: dict[str, float] = Field(default_factory=dict)
    tension_baseline_var: dict[str, float] = Field(default_factory=dict)
    tension_baseline_n: dict[str, int] = Field(default_factory=dict)
    # This tick's admitted-deviation scalar (orion.attention.tension.
    # competition.deviation_pressure), in [0, 1]. 0.0 is a real "nothing
    # admitted this tick" reading, never a fabricated absence -- see that
    # function's own docstring. Consumed by orion.field.pressure.
    # field_pressures() as the "deviation_pressure" PRESSURE_DIMENSIONS entry.
    tension_deviation_pressure: float = 0.0
    # This tick's Borda-winning target (orion.attention.tension.competition
    # .TickResult.winner) -- a node_id (e.g. "node:athena"), or None on a
    # quiet tick where nothing was admitted. Deliberately NOT wired into
    # orion/proposals/builder.py's target_binding when tension_deviation_
    # pressure first shipped (docs/superpowers/specs/2026-08-16-tension-
    # driven-mutating-dispatch-design.md's own non-goals -- no real consumer
    # yet, and CLAUDE.md 0A's keyword-cathedral rule requires one in the same
    # patch that adds a new concept). Added now because a real consumer
    # exists: Hub's endogenous-outreach trigger (services/orion-hub/scripts/
    # tension_outreach_trigger.py) needs identity, not just magnitude, to
    # honestly claim "I noticed X changing" rather than a content-free "I'm
    # reaching out" wrapped around an unrelated number.
    tension_borda_winner_target_id: str | None = None
    # Level-aware significance (2026-08-18, docs/superpowers/specs/2026-08-16-
    # level-aware-significance-design.md): the axis `tension_deviation_pressure`
    # structurally cannot see. DeviationGate is a change-detector -- a channel
    # steadily overloaded for hours re-centers its own EWMA baseline and reads
    # calm, by design. `orion.field.significance.sustained_load_pressure`
    # instead reads `orion.field.regime.channel_regime()` (level + dispersion
    # as SEPARATE axes, no adaptive baseline at all) over a real window, and
    # only counts a channel that is currently BOTH high level AND low
    # dispersion (`loaded_steady`) -- not `loaded_volatile`, which a
    # change-detector or the reconstruction-loss anomaly scorer
    # (app/anomaly_scorer.py) can plausibly already catch; voting on that too
    # would blur this metric back into redundancy with what already exists
    # (see significance.py's own module docstring for the full independence
    # argument). 0.0 is a real "nothing loaded_steady right now" reading,
    # never a fabricated absence -- same convention `tension_deviation_
    # pressure` uses for its own quiet-tick 0.0. Consumed by orion.field.
    # pressure.field_pressures() as the "sustained_load_pressure"
    # PRESSURE_DIMENSIONS entry. Written by services/orion-field-digester/
    # app/digestion/significance.py, throttled (not every hot tick -- the
    # window read is a real Postgres query, not an O(1) EWMA update) via
    # `sustained_load_computed_at` below.
    sustained_load_pressure: float = 0.0
    # When `sustained_load_pressure` was last actually RECOMPUTED (not just
    # carried forward unchanged from the prior tick). Compared against `now`
    # each hot tick to decide whether this tick's real window-read is due yet
    # -- persisted here (round-tripped through Postgres every tick like
    # `tension_baseline_mu` above) rather than held on the worker process
    # instance, so the throttle survives a restart instead of firing on every
    # tick for the first `check_interval_sec` after one.
    sustained_load_computed_at: datetime | None = None
    topology_id: str | None = None
    topology_version: str | None = None
    topology_loaded_from: str | None = None
