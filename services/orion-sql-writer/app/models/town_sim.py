"""Isolated simulation artifacts; never generic chat or memory records."""
from sqlalchemy import Column, DateTime, Float, Integer, BigInteger, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.db import Base

class TownSimSceneSQL(Base):
    __tablename__='town_sim_scene_resolved'
    id=Column(String,primary_key=True)
    world_id=Column(String,nullable=False)
    scene_id=Column(String,nullable=False)
    participants=Column(JSONB,nullable=False)
    purpose=Column(String,nullable=False)
    legal_results=Column(JSONB,nullable=False)
    chosen_results=Column(JSONB,nullable=False)
    effects_applied=Column(JSONB,nullable=False)
    seed=Column(BigInteger,nullable=False)
    status=Column(String,nullable=False)
    transcript=Column(JSONB,nullable=False)
    created_at=Column(DateTime(timezone=True),server_default=func.now())

class TownSimPredictionSQL(Base):
    __tablename__='town_sim_prediction_score'
    id=Column(String,primary_key=True)
    world_id=Column(String,nullable=False)
    target_id=Column(String,nullable=False)
    purpose=Column(String,nullable=False)
    probabilities=Column(JSONB,nullable=False)
    outcome=Column(JSONB,nullable=False)
    brier=Column(Float,nullable=False)
    brier_resample=Column(Float)
    brier_base_rate=Column(Float,nullable=False)
    baseline_count=Column(Integer,nullable=False)
    resample_count=Column(Integer,nullable=False)
    status=Column(String,nullable=False)
    signal_status=Column(String,nullable=False)
    created_at=Column(DateTime(timezone=True),server_default=func.now())
