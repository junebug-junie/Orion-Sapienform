from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import json
from pydantic import BaseModel


class SparkIntrospectionInput(BaseModel):
    """
    Wire-level payload for Spark introspection logs.

    This is what SQL writer receives over the bus and normalizes
    before writing to the DB.
    """
    id: Optional[str] = None
    trace_id: str
    source: str
    kind: Optional[str] = None

    prompt: Optional[str] = None
    response: Optional[str] = None

    introspection: str  # required: the internal note
    spark_meta: Optional[Dict[str, Any]] = None

    created_at: Optional[datetime] = None

    def normalize(self) -> "SparkIntrospectionInput":
        """
        Fill in defaults and ensure types are write-friendly.
        """
        # For MVP: assume at most one introspection per trace, so use trace_id.
        # If you later want multiple notes per trace, you can change this to
        # append a suffix or use a uuid.
        if self.id is None:
            self.id = self.trace_id

        if self.created_at is None:
            self.created_at = datetime.utcnow()

        # Ensure spark_meta will serialize cleanly
        if isinstance(self.spark_meta, dict):
            # We keep it as dict here; the worker will json.dumps before insert.
            pass

        return self
