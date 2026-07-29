from datetime import datetime

from sqlalchemy import Column, DateTime, JSON, String
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base

# Generic JSON on SQLite (tests), JSONB on Postgres (prod) -- same pattern as
# app/models/spark_telemetry.py.
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class HarnessTurnTraceSQL(Base):
    """Durable, one-row-per-turn capture of the unified turn's finalize chain
    (5a substrate appraisal, 5b reflection/verdict, 6b outcome, 7 closure).

    All four molecules already PUBLISH unredacted on the bus today
    (orion/harness/finalize.py) -- this table just stops throwing them away.
    Each of the four source kinds (harness.run.v1, harness.verdict.molecule.v1,
    harness.turn.outcome.v1, harness.post_turn.closure.v1) arrives as its own
    bus message and fills in one JSONB column via upsert-by-correlation_id
    (see app/harness_turn_trace_persist.py); a turn's row is complete once all
    four have landed, partial before that -- never blocks on the others.
    """

    __tablename__ = "harness_turn_trace"

    correlation_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # HarnessRunV1: draft_text, substrate_appraisal, reflection, verdict_molecule_id,
    # finalize_ran/changed, quick_lane_skipped_5b, step_count, exit_code,
    # compliance_verdict, grounding_status, grammar_event_ids, final_text.
    run_artifact = Column(_JSONB, nullable=True)
    # HarnessVerdictMoleculeV1: reflection (duplicated from run_artifact) + cortex_trace_id.
    verdict_molecule = Column(_JSONB, nullable=True)
    # HarnessTurnOutcomeMoleculeV1: final_hash, surprise_resolved, finalize_failed,
    # failure_reason, draft_text_excerpt.
    outcome_molecule = Column(_JSONB, nullable=True)
    # HarnessPostTurnClosureV1: surprise_unresolved, referent fields (user_message_excerpt,
    # stance_imperative) when surprise was unresolved.
    closure = Column(_JSONB, nullable=True)
