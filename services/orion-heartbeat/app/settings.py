from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    service_name: str = Field("orion-heartbeat", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: Optional[str] = Field(None, alias="NODE_NAME")
    instance_id: Optional[str] = Field(None, alias="INSTANCE_ID")

    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    health_channel: str = Field("orion:system:health", alias="ORION_HEALTH_CHANNEL")

    # v0 is additive/read-only: subscribes to the existing, already-standardized
    # grammar stream (see design doc, "Current architecture" -- GrammarEventV1 +
    # orion-substrate-runtime already solved the per-organ ingestion problem the
    # 2026-05-01 heartbeat charter stalled on). No new channel is registered;
    # this is a second, additive consumer of a channel that already exists.
    channel_grammar_event: str = Field("orion:grammar:event", alias="CHANNEL_GRAMMAR_EVENT")

    # H1 (boundary/bulk entanglement) is computed periodically, not on every
    # absorbed atom -- entropy_profile() is cheap (~6ms) but there's no reason
    # to recompute it more often than an operator/debug surface would ever
    # read it.
    h1_interval_sec: float = Field(30.0, alias="HEARTBEAT_H1_INTERVAL_SEC")

    # Deterministic substrate seed (initial random MPS state before any
    # absorption) -- fixed by default so a restart without a snapshot
    # reproduces the same starting point; v0 has no crash-safe persistence
    # (charter's own snapshot/restore machinery is explicitly deferred, see
    # design doc "Explicit deferrals"). Now the BASE seed for the N-trajectory
    # ensemble (trajectory i uses substrate_seed + i) -- see EnsembleSubstrate.
    substrate_seed: int = Field(42, alias="HEARTBEAT_SUBSTRATE_SEED")

    http_port: int = Field(7251, alias="HEARTBEAT_HTTP_PORT")

    # --- N-trajectory dissipation ensemble (2026-07-28) ---
    # docs/superpowers/specs/2026-07-28-precision-weighted-attention-organ-
    # and-heartbeat-discrimination-design.md. Defaults below are the
    # sweep-derived values from scripts/analysis/measure_heartbeat_ensemble_
    # calibration.py, confirmed against both synthetic and real historical
    # `grammar_events` replay -- not guessed. Operator-tunable via env,
    # matching this file's own existing pattern (HEARTBEAT_H1_INTERVAL_SEC),
    # and re-validatable at any time by re-running that harness script.

    # N=8: harness measured ~118ms/tick compute cost, comfortably under this
    # system's real average tick interval (~300ms, from live tick_count/
    # uptime) with headroom for bursts. Not yet re-validated for how the
    # estimate stabilizes as N increases (Missing Question 11) -- a starting
    # default, not a converged optimum.
    n_trajectories: int = Field(8, alias="HEARTBEAT_N_TRAJECTORIES")

    # Fraction each relaxation-gate application contracts a site toward this
    # trajectory's own seed-derived target (see mps_state.py's
    # compute_relaxation_targets).
    decay_gamma: float = Field(0.2, alias="HEARTBEAT_DECAY_GAMMA")

    # Base per-site decay probability before spread-gating (ensemble.py's
    # spread_gate) scales it down when trajectories disagree.
    base_decay_prob: float = Field(0.15, alias="HEARTBEAT_BASE_DECAY_PROB")

    # How sharply disagreement among trajectories suppresses decay --
    # effective_decay_prob = base_decay_prob * exp(-sensitivity * spread).
    decay_spread_sensitivity: float = Field(4.0, alias="HEARTBEAT_DECAY_SPREAD_SENSITIVITY")

    # Two-site entangling reheat gate strength (rotation angle) -- see
    # mps_state.py's module docstring for why reheat must be two-site, not
    # one-site (a one-site gate is provably incapable of moving entanglement
    # entropy across any cut).
    reheat_strength: float = Field(0.08, alias="HEARTBEAT_REHEAT_STRENGTH")

    # Scales real live bus_synaptic activity (see falkordb_* settings below)
    # into a per-bond reheat probability: reheat_prob = reheat_prob_scale *
    # min(1, raw_mean_abs_gap_zscore / BUS_SYNAPTIC_ZSCORE_SATURATION).
    reheat_prob_scale: float = Field(0.02, alias="HEARTBEAT_REHEAT_PROB_SCALE")

    # Wall-clock cadence for the decay/reheat dissipation loop -- deliberately
    # a SEPARATE timer from message consumption (Missing Question 7,
    # resolved this way): if organs go quiet, real messages stop arriving,
    # but dissipation must keep running on its own clock or "rest" could
    # never actually be reached.
    decay_reheat_interval_sec: float = Field(2.0, alias="HEARTBEAT_DECAY_REHEAT_INTERVAL_SEC")

    # FalkorDB connection for the real bus_synaptic reheat driver -- same
    # naming convention as services/orion-substrate-runtime's .env_example
    # (FALKORDB_URI / FALKORDB_BUS_GRAPH), a second, additive, read-only
    # consumer of the same already-live graph, no new graph/schema.
    falkordb_uri: str = Field("redis://orion-athena-falkordb:6379", alias="FALKORDB_URI")
    falkordb_bus_graph: str = Field("orion_bus_synapse", alias="FALKORDB_BUS_GRAPH")

    # Bound on the intake->absorb queue (see service.py's _run()/
    # _absorb_worker_loop() split) -- decouples cheap message intake from
    # the N-trajectory absorb() cost so a burst doesn't stall the bus
    # consumer loop itself, only the ensemble's freshness. Bounded (not
    # unbounded) so a sustained worker stall degrades to dropped-and-counted
    # events rather than unbounded memory growth.
    absorb_queue_maxsize: int = Field(10_000, alias="HEARTBEAT_ABSORB_QUEUE_MAXSIZE")


settings = Settings()
