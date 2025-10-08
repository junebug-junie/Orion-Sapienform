from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from orion.core.sql_router.db import get_db

def build_writer_router(model_class, db_writer_func):
    router = APIRouter()

    @router.post("/write")
    def write(payload: dict, db: Session = Depends(get_db)):
        return db_writer_func(model_class, payload, db)

    return router
