from __future__ import annotations

import pytest

from app.callable_namespace import ContextNamespace
from app.security import PolicyBlockedError
from orion.schemas.context_exec import ContextExecPermissionV1


def test_no_write_policy_blocks_mutations() -> None:
    perms = ContextExecPermissionV1()
    ns = ContextNamespace(permissions=perms)
    with pytest.raises(PolicyBlockedError):
        ns.memory.write("x")
    with pytest.raises(PolicyBlockedError):
        ns.repo.write("path", "data")
    with pytest.raises(PolicyBlockedError):
        ns._guard("shell")
    with pytest.raises(PolicyBlockedError):
        ns._guard("network")
