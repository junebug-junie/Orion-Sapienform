"""Bus names shared by the orion-introspect tool and its owning-service responders."""
from __future__ import annotations

DREAM_REQUEST_CHANNEL = "orion:introspect:dream:request"
CURIOSITY_REQUEST_CHANNEL = "orion:introspect:curiosity:request"
ORION_DAY_REQUEST_CHANNEL = "orion:introspect:orion_day:request"
RECORD_CORRECTION_REQUEST_CHANNEL = "orion:introspect:record_correction:request"
NODE_HISTORY_REQUEST_CHANNEL = "orion:introspect:node_history:request"
QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL = "orion:introspect:queue_curiosity_focus:request"
RESULT_PREFIX = "orion:introspect:result:"
REQUEST_KIND = "introspect.tool.request.v1"
RESULT_KIND = "introspect.tool.result.v1"
