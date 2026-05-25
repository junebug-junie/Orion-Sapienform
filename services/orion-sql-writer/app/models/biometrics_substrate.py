from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class SubstrateReductionCursorSQL(Base):
    __tablename__ = "substrate_reduction_cursor"

    cursor_name = Column(String, primary_key=True)
    last_event_created_at = Column(DateTime(timezone=True), nullable=True)
    last_event_id = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class SubstrateNodeBiometricsProjectionSQL(Base):
    __tablename__ = "substrate_node_biometrics_projection"

    projection_id = Column(String, primary_key=True)
    generated_at = Column(DateTime(timezone=True), nullable=False)
    projection_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class SubstrateActiveNodePressureProjectionSQL(Base):
    __tablename__ = "substrate_active_node_pressure_projection"

    projection_id = Column(String, primary_key=True)
    generated_at = Column(DateTime(timezone=True), nullable=False)
    projection_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class SubstrateOrganEmissionSQL(Base):
    __tablename__ = "substrate_organ_emissions"

    emission_id = Column(String, primary_key=True)
    organ_id = Column(String, nullable=False)
    invocation_id = Column(String, nullable=False)
    emission_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("idx_substrate_organ_emissions_created", "created_at"),)


class SubstrateReductionReceiptSQL(Base):
    __tablename__ = "substrate_reduction_receipts"

    receipt_id = Column(String, primary_key=True)
    organ_id = Column(String, nullable=True)
    emission_id = Column(String, nullable=True)
    receipt_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("idx_substrate_reduction_receipts_created", "created_at"),)
