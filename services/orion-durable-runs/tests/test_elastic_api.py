"""The internal elastic API is tokenless but retains operational gates."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
import pytest

@pytest.mark.parametrize("shadow,eligible,code",[(True,True,503),(False,False,409),(False,True,200)])
def test_tokenless_target_retains_shadow_and_eligibility_gates(monkeypatch,shadow,eligible,code):
    monkeypatch.setenv("POSTGRES_URI","postgresql://unused/unused")
    from app import main
    @asynccontextmanager
    async def transaction():
        yield object()
    store=SimpleNamespace(row=AsyncMock(return_value={"desired_target":"diffusion","run_id":None}),
        intent=AsyncMock(),snapshot=AsyncMock(return_value={"desired_target":"agent-burst"}))
    runtime=SimpleNamespace(settings=SimpleNamespace(elastic_shadow=shadow),
        elastic=SimpleNamespace(environment=AsyncMock(return_value={"eligible":eligible}),store=store),
        store=SimpleNamespace(transaction=transaction),_wake=asyncio.Event())
    monkeypatch.setattr(main,"_admission",lambda:runtime)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url="http://authority") as client:
            response=await client.post('/elastic/target',json={"target":"agent-burst"})
            assert response.status_code==code
        if code==200:
            store.intent.assert_awaited_once()
            assert runtime._wake.is_set()
        else:
            store.intent.assert_not_called()
    asyncio.run(scenario())
