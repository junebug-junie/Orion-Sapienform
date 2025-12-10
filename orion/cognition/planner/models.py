# orion/cognition/planner/models.py
from typing import Optional
from pydantic import BaseModel


class VerbConfig(BaseModel):
    name: str
    description: Optional[str] = None
    group: Optional[str] = None
    # maybe:
    # input_schema: Optional[dict] = None
    # output_schema: Optional[dict] = None
    # examples: Optional[List[str]] = None
    # tags: Optional[List[str]] = None
