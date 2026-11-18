"""Dream-side configuration of the shared Curiosity compaction store contract."""
from functools import lru_cache
from orion.curiosity.compaction_store import Store
from app.settings import settings

@lru_cache(maxsize=1)
def store():
    return Store(settings.POSTGRES_URI)

def recent(limit=20):
    return store().recent(limit)
