# tests/test_ai_system.py
import pytest
import asyncio
from emergence.core.ai_system import EmergentAISystem

@pytest.mark.asyncio
async def test_basic_step():
    system = EmergentAISystem()
    await system.step()
    keys = system.memory.last_keys()
    assert "vision" in keys
    assert "introspection" in keys
    assert "rdf" in keys

