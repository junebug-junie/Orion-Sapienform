from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from .settings import settings


engine = create_engine(settings.database_url, pool_pre_ping=True)
session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Session = scoped_session(session_factory)
Base = declarative_base()


def get_session():
    return Session()


def remove_session() -> None:
    Session.remove()
