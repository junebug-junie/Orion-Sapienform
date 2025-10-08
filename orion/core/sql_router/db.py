from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

POSTGRES_URI = os.getenv("POSTGRES_URI", "sqlite:///./default.db")

Base = declarative_base()
engine = create_engine(POSTGRES_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_models(model_classes: list):
    for model in model_classes:
        model.__table__.create(bind=engine, checkfirst=True)
