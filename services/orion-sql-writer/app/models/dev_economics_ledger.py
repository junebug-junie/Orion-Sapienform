from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class DevEconomicsLedgerSQL(Base):
    """Durable per-tick record of orion-cocreation-signals' dev_economics
    producer (docs/superpowers/specs/2026-07-30-dev-economics-signal-
    design.md) -- the real GROWTH in Claude Code token/word/$-cost totals
    since the producer's last check, published on
    ``orion:substrate:dev_economics_ledger``.

    First real consumer of that channel (previously ``consumer_services: []``
    -- pure shadow write, docs/superpowers/pr-reports/2026-08-12-dev-
    economics-hub-observability.md) -- exists specifically so the Hub's
    cocreation-signals tab has real rows to read instead of an empty table.

    ``event_id`` (``DevEconomicsLedgerV1.event_id``) is the primary key so a
    redelivered event upserts idempotently, same pattern as
    ``DominanceStreakTickSQL.tick_telemetry_id``. Append-only, unconditional
    (not gated on any threshold) -- same "complete history, not just notable
    moments" reasoning as ``substrate_codebase_delta_log``.
    """

    __tablename__ = "dev_economics_ledger_log"

    event_id = Column(String, primary_key=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    # window_since/window_until deliberately have no column: both are always
    # set to observed_at on the wire (DevEconomicsLedgerV1's own docstring --
    # bookkeeping fields only, kept for schema shape consistency with the
    # other windowed signals, not a real window boundary for this diff-based
    # producer). Persisting two columns that always equal observed_at would
    # be pure redundancy, not real data.

    session_count = Column(Integer, nullable=False)
    subagent_transcript_count = Column(Integer, nullable=False)

    total_input_tokens = Column(Integer, nullable=False)
    total_output_tokens = Column(Integer, nullable=False)
    total_cache_creation_input_tokens = Column(Integer, nullable=False)
    total_cache_read_input_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)

    total_assistant_word_count = Column(Integer, nullable=False)
    total_human_word_count = Column(Integer, nullable=False)

    # Nullable -- a real "nothing priceable this tick" fact, never a
    # fabricated 0.0. See DevEconomicsLedgerV1's own docstring.
    total_estimated_cost_usd = Column(Float, nullable=True)
    priced_session_count = Column(Integer, nullable=False, default=0)
    unpriced_session_count = Column(Integer, nullable=False, default=0)

    # Real model->count histogram, e.g. {"claude-sonnet-5": 3} -- stored as
    # the raw JSON text; the Hub route parses it, same pattern as
    # substrate_codebase_delta_log's own event_json handling.
    model_mix_json = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
