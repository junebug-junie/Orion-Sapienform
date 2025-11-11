from sqlalchemy import Column, String, JSON
from uuid import uuid4
from orion.core.sql_router.db import Base
from app.settings import settings

def generate_biometrics_model():
    """
    Returns a SQLAlchemy model with the table name defined in .env → TABLE_NAME
    """
    return type(
        "BiometricsSQL",
        (Base,),
        {
            "__tablename__": settings.TABLE_NAME,
            "id": Column(String, primary_key=True, default=lambda: str(uuid4())),
            "node": Column(String),
            "timestamp": Column(String),
            "gpu": Column(JSON),
            "cpu": Column(JSON),
        },
    )

BiometricsSQL = generate_biometrics_model()
