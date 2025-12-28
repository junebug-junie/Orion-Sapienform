import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from app.settings import settings

engine = create_engine(
    settings.postgres_uri,
    pool_pre_ping=True,
    json_serializer=json.dumps,
    json_deserializer=json.loads,
)

# Create a session factory
session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a thread-safe, scoped session registry
Session = scoped_session(session_factory)

Base = declarative_base()


def get_session():
    return Session()

def remove_session():
    """Removes the current thread-local session."""
    Session.remove()
