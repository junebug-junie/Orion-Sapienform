from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from orion.schemas.cortex.contracts import CortexChatRequest, CortexClientResult

# In the merge, the upstream branch introduced this file `orion/schemas/cortex/gateway.py`
# which seems to duplicate `CortexChatRequest` definition which I added to `contracts.py`.
# To resolve, I will re-export the one from `contracts.py` or keep it here if that's the canonical place.
# The user instruction was to "canonicalize it into orion/schemas/*".
# `contracts.py` seems like a good place, but if upstream added `gateway.py`, I should probably respect that structure
# OR alias it.

# However, `contracts.py` is where `CortexClientResult` lives.
# `gateway.py` in upstream imports `CortexClientResult` from `contracts.py`.

# I will keep `CortexChatResult` here as it is new from upstream.
# And I will alias `CortexChatRequest` to the one in `contracts.py` to avoid duplication.

class CortexChatResult(BaseModel):
    cortex_result: CortexClientResult = Field(..., description="The raw result from Cortex Orchestrator")
    final_text: Optional[str] = Field(default=None, description="Convenience field for the final text response")
