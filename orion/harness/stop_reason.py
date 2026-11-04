"""Why a harness turn stopped, as one string on its record (``HarnessRunV1.stop_reason``).

Before this, a killed turn left only ``exit_code=-9`` and no ``final_text``: nobody could tell a
GPU-pool recall or Juniper's stop button from Orion failing. orion-prediction then voided the turn's
self-bets ``failed_to_execute`` and the self-monitor counted it as ``turn.run`` not running.

Values:
    ``cancel:<reason>``   a harness cancel (HarnessRunCancelV1) killed the motor's process; ``<reason>``
                          is the cancel's own reason (``user_stop``, ``client_disconnect``, ...).
    ``<motor error code>`` the motor ended the turn itself or the CLI failed (``fcc_timeout``,
                          ``fcc_stream_stalled``, ``fcc_context_ceiling_exceeded``, ``fcc_nonzero_exit``...).
    ``None``              the motor raised no error, or the record predates this field.

Only a cancel whose reason is in ``OUTSIDE_CANCEL_REASONS`` counts as stopped from outside. An unknown
cancel reason, a motor code, or no value keeps the turn a real failure: unknown never loosens.
"""

from __future__ import annotations

CANCEL_PREFIX = "cancel:"

# Every reason a producer sends on orion:harness:run:cancel today. Add a new one here when a new
# producer starts sending it; until then its turns stay ``failed_to_execute``.
OUTSIDE_CANCEL_REASONS = frozenset(
    {
        "user_stop",                # Hub stop button (api_routes.api_chat_turn_cancel)
        "client_disconnect",        # Hub websocket closed mid-turn
        "ws_error",                 # Hub websocket handler crashed mid-turn
        "durable_attempt_stopped",  # durable-runs stopped the attempt (hold lost/recalled, run control)
        "worker_recovery",          # durable-runs recovering a dead worker's turn
    }
)
# Deliberately NOT outside: "durable_attempt_timeout" (the attempt's own time budget ran out -- the
# turn was too slow, like the motor's fcc_timeout).


def turn_stop_reason(*, cancel_reason: str | None, error_code: str | None) -> str | None:
    """The cancel reason if a cancel killed the turn, else the motor's error code, else None."""
    cancel = str(cancel_reason or "").strip()
    if cancel:
        return f"{CANCEL_PREFIX}{cancel}"
    code = str(error_code or "").strip()
    return code or None


def is_outside_stop(stop_reason: object) -> bool:
    """True only for a known outside cancel. Anything else -- a motor code, an unknown cancel reason,
    None, a non-string -- is False."""
    if not isinstance(stop_reason, str) or not stop_reason.startswith(CANCEL_PREFIX):
        return False
    return stop_reason[len(CANCEL_PREFIX):] in OUTSIDE_CANCEL_REASONS
