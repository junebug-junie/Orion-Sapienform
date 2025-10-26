# In services/orion-sql-writer/app/db_utils.py

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from app.db import get_session, remove_session

def get_model_for_table(table_name: str):
    model = MODEL_MAP.get(table_name)
    if not model:
        raise ValueError(f"No ORM model registered for '{table_name}'")
    return model
