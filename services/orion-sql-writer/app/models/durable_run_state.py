from sqlalchemy import JSON, Column, DateTime, Index, String
from sqlalchemy.sql import func

from app.db import Base


class DurableRunStateSQL(Base):
    """One row per node transition of a durable cognition run
    (DurableRunStateV1, orion/schemas/durable_run.py). Single writer:
    orion-sql-writer off `orion:durable:run:state`. The runner's own
    checkpoint tables are its private storage; this is the queryable
    history -- which runs resumed, from which node, how many were abandoned.
    """

    __tablename__ = "substrate_durable_run_state"

    entry_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    generated_at = Column(DateTime(timezone=True), nullable=False)
    run_id = Column(String, nullable=False)
    workflow = Column(String, nullable=False)
    thread_id = Column(String, nullable=False)
    node = Column(String, nullable=False)
    next_node = Column(String, nullable=True)
    status = Column(String, nullable=False)
    resumed_from_node = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)
    detail = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_substrate_durable_run_state_run_created", "run_id", "created_at"),
    )
