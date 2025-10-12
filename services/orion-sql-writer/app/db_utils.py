from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from app.db import get_session
from app.models import CollapseEnrichment, CollapseMirror

MODEL_MAP = {
    "collapse_enrichment": CollapseEnrichment,
    "collapse_mirror": CollapseMirror,
}

def get_model_for_table(table_name: str):
    model = MODEL_MAP.get(table_name)
    if not model:
        raise ValueError(f"No ORM model registered for '{table_name}'")
    return model


def upsert_record(table_name: str, data: dict):
    model_cls = get_model_for_table(table_name)
    session = get_session()
    try:
        stmt = insert(model_cls).values(**data)
        update_cols = {c.name: c for c in stmt.excluded if c.name != model_cls.__table__.primary_key.columns.keys()[0]}
        stmt = stmt.on_conflict_do_update(
            index_elements=model_cls.__table__.primary_key.columns.keys(),
            set_=update_cols,
        )
        session.execute(stmt)
        session.commit()
        print(f"✅ Upserted {table_name}:{data.get('id')}")
    except SQLAlchemyError as e:
        session.rollback()
        print(f"❌ Upsert failed for {table_name}:{data.get('id')} — {e}")
    finally:
        session.close()
