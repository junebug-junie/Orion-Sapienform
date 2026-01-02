from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from orion.schemas.cortex.contracts import CortexChatRequest

# Re-exporting CortexChatRequest for backward compatibility inside the app,
# but it is now sourced from the shared contract schema.
