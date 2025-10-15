from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .settings import settings

# Create the SQLAlchemy engine that connects to your PostgreSQL database
# using the URI from your settings file.
engine = create_engine(settings.POSTGRES_URI)

# Create a configured "Session" class. This is the factory for new sessions.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# A base class for your SQLAlchemy models to inherit from.
Base = declarative_base()

def get_db():
    """
    A dependency for FastAPI routes that provides a database session and
    ensures it's closed after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

