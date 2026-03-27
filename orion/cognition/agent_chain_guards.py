"""Pure guard logic for agent-chain runtime (Pass 2 testability)."""


def triage_must_finalize(*, tool_id: str, step_idx: int, prior_trace_len: int) -> bool:
    """Triage is disallowed once any prior delegated step exists."""
    if tool_id != "triage":
        return False
    return step_idx > 0 or prior_trace_len > 0


def repeated_plan_action_needs_delivery(*, tool_id: str, tools_called: list[str]) -> bool:
    """Second or later plan_action in the same chain should become a delivery verb."""
    return tool_id == "plan_action" and "plan_action" in tools_called


def consecutive_tool_count(*, tools_called: list[str], candidate: str) -> int:
    """Count consecutive calls of `candidate` from the tail of tools_called."""
    count = 0
    for tool in reversed(tools_called):
        if tool != candidate:
            break
        count += 1
    return count


def plan_action_saturated(*, tool_id: str, tools_called: list[str], max_consecutive: int = 2) -> bool:
    """True when a new plan_action would exceed max_consecutive repeated plan_action calls."""
    if tool_id != "plan_action":
        return False
    return consecutive_tool_count(tools_called=tools_called, candidate="plan_action") >= max_consecutive
