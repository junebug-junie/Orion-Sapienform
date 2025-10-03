from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from orion.schemas.collapse_mirror import CollapseMirrorEntry
from app.db import get_db
from app.services.collapse_service import log_and_persist

router = APIRouter()

# 🪞 Collapse Mirror: Log new entry
@router.post("/log/collapse")
def log_collapse(entry: CollapseMirrorEntry, db: Session = Depends(get_db)):
    return log_and_persist(entry, db)
