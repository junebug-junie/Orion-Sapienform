import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from app.settings import settings


def build_engine_connect_args(
    statement_timeout_ms: int,
    *,
    lock_timeout_ms: int = 0,
) -> dict:
    """Postgres session options applied to every sql-writer connection."""
    parts: list[str] = []
    if statement_timeout_ms > 0:
        parts.append(f"statement_timeout={statement_timeout_ms}")
    if lock_timeout_ms > 0:
        parts.append(f"lock_timeout={lock_timeout_ms}")
    if not parts:
        return {}
    return {"options": " ".join(f"-c {part}" for part in parts)}


engine = create_engine(
    settings.postgres_uri,
    pool_pre_ping=True,
    pool_size=max(1, int(settings.sql_writer_db_pool_size)),
    max_overflow=max(0, int(settings.sql_writer_db_max_overflow)),
    connect_args=build_engine_connect_args(
        settings.sql_writer_db_statement_timeout_ms,
        lock_timeout_ms=settings.sql_writer_db_lock_timeout_ms,
    ),
    json_serializer=json.dumps,
    json_deserializer=json.loads,
)

grammar_engine = create_engine(
    settings.postgres_uri,
    pool_pre_ping=True,
    pool_size=max(1, int(settings.sql_writer_grammar_pool_size)),
    max_overflow=max(0, int(settings.sql_writer_grammar_pool_max_overflow)),
    connect_args=build_engine_connect_args(
        settings.sql_writer_grammar_statement_timeout_ms,
        lock_timeout_ms=settings.sql_writer_grammar_lock_timeout_ms,
    ),
    json_serializer=json.dumps,
    json_deserializer=json.loads,
)

# Create a session factory
session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
grammar_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=grammar_engine)

# Create a thread-safe, scoped session registry
Session = scoped_session(session_factory)
GrammarSession = scoped_session(grammar_session_factory)

Base = declarative_base()


def get_session():
    return Session()


def remove_session():
    """Removes the current thread-local session."""
    Session.remove()


def get_grammar_session():
    return GrammarSession()


def remove_grammar_session():
    """Removes the current thread-local grammar session."""
    GrammarSession.remove()


def dispose_grammar_pool() -> None:
    """Drop all grammar-pool connections (e.g. after cancel/timeout)."""
    GrammarSession.remove()
    grammar_engine.dispose()
