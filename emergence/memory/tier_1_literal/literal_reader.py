# emergence/memory/tier1_literal/literal_reader.py
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from emergence.memory.tier1_literal.models import MemoryEntry, get_engine

def read_recent_literal_entries(limit=5) -> list:
    """
    Fetch the N most recent literal memory entries from Postgres.
    """
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(MemoryEntry).order_by(desc(MemoryEntry.timestamp)).limit(limit)
        results = session.execute(stmt).scalars().all()
        return [r.to_dict() for r in results]

