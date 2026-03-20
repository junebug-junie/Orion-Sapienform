"""Pure guard logic for agent-chain runtime (Pass 2 testability)."""


def triage_must_finalize(*, tool_id: str, step_idx: int, prior_trace_len: int) -> bool:
    """Triage is disallowed once any prior delegated step exists."""
    if tool_id != "triage":
        return False
    return step_idx > 0 or prior_trace_len > 0


def repeated_plan_action_needs_delivery(*, tool_id: str, tools_called: list[str]) -> bool:
    """Second or later plan_action in the same chain should become a delivery verb."""
    return tool_id == "plan_action" and "plan_action" in tools_called
