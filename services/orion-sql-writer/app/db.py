import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.settings import settings

# Engine w/ JSON helpers + pre-ping for long-lived services
engine = create_engine(
    settings.POSTGRES_URI,
    pool_pre_ping=True,
    json_serializer=json.dumps,
    json_deserializer=json.loads,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_session():
    return SessionLocal()
