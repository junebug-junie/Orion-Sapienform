import asyncio
from unittest.mock import AsyncMock
import pytest
from pydantic import ValidationError
from test_api import main_module
from orion.schemas.gpu_slot import GpuSlotRequestV1

gpu = main_module.gpu2

def req(target="agent-burst"):
    return GpuSlotRequestV1(slot="circe-gpu2", target=target,operation_id="test:1",generation=1)

def snapshot(active="diffusion", state="running"):
    return {"active":active,"targets":{key:{"state":state if key == active else "exited",
        "containers":[{"state":state}] if key == active else []} for key in ("diffusion","agent-burst")}}

@pytest.mark.parametrize("slot,target", [("unknown","agent"),("circe-gpu1","diffusion"),("circe-gpu2","affect")])
def test_fixed_pair_contract(slot,target):
    with pytest.raises(ValidationError):
        GpuSlotRequestV1(slot=slot,target=target,operation_id="x",generation=1)

def test_unknown_command_fields_forbidden():
    with pytest.raises(ValidationError):
        GpuSlotRequestV1(**req().model_dump(),compose_service="atlas-chat")

def test_drain_before_stop_before_model_ready(monkeypatch):
    calls=[]
    async def record(value): calls.append(value)
    monkeypatch.setattr(gpu,"authority",AsyncMock(return_value={"can_transition":True}))
    monkeypatch.setattr(gpu,"status",AsyncMock(return_value=snapshot()))
    monkeypatch.setattr(gpu,"drain_diffusion",lambda:record("drain"))
    monkeypatch.setattr(gpu,"stop",lambda target:record("stop:"+target))
    monkeypatch.setattr(gpu,"start",lambda target:record("ready:"+target))
    result=asyncio.run(gpu.transition(req()))
    assert result["status"] == "success"
    assert calls == ["drain","stop:diffusion","ready:agent-burst"]

@pytest.mark.parametrize("state",["paused","restarting","unknown"])
def test_uncertain_container_never_starts_other(state,monkeypatch):
    monkeypatch.setattr(gpu,"authority",AsyncMock(return_value={"can_transition":True}))
    monkeypatch.setattr(gpu,"status",AsyncMock(return_value=snapshot(state=state)))
    start=AsyncMock();monkeypatch.setattr(gpu,"start",start)
    assert asyncio.run(gpu.transition(req()))["status"] == "failed"
    start.assert_not_called()

def test_busy_generation_timeout_only_unlatches_no_stop(monkeypatch):
    monkeypatch.setattr(gpu,"authority",AsyncMock(return_value={"can_transition":True}))
    monkeypatch.setattr(gpu,"status",AsyncMock(return_value=snapshot()))
    monkeypatch.setattr(gpu,"drain_diffusion",AsyncMock(side_effect=RuntimeError("diffusion_drain_timeout")))
    stop=AsyncMock();start=AsyncMock();resume=AsyncMock()
    monkeypatch.setattr(gpu,"stop",stop);monkeypatch.setattr(gpu,"start",start);monkeypatch.setattr(gpu,"request",resume)
    assert asyncio.run(gpu.transition(req()))["status"] == "failed"
    stop.assert_not_called();start.assert_not_called()
    assert resume.call_args.args[1] == {"draining":False}

def test_failed_start_restores_and_failed_restore_observable(monkeypatch):
    monkeypatch.setattr(gpu,"authority",AsyncMock(return_value={"can_transition":True}))
    monkeypatch.setattr(gpu,"status",AsyncMock(return_value=snapshot()))
    monkeypatch.setattr(gpu,"drain_diffusion",AsyncMock())
    monkeypatch.setattr(gpu,"stop",AsyncMock())
    start=AsyncMock(side_effect=[RuntimeError("load_failed"),None]);monkeypatch.setattr(gpu,"start",start)
    result=asyncio.run(gpu.transition(req()))
    assert result["restored"] is True
    assert [c.args[0] for c in start.call_args_list] == ["agent-burst","diffusion"]
    start.side_effect=RuntimeError("load_failed")
    assert "restoration_failed" in asyncio.run(gpu.transition(req()))["error"]

def test_noop_and_independent_slot_lock(monkeypatch):
    async def scenario():
        monkeypatch.setattr(gpu.settings,"GPU2_ENABLED",True)
        monkeypatch.setattr(gpu,"authority",AsyncMock(return_value={"can_transition":True}))
        monkeypatch.setattr(gpu,"status",AsyncMock(return_value=snapshot(active="agent-burst")))
        monkeypatch.setattr(gpu,"model_ready",AsyncMock(return_value=True))
        start=AsyncMock();monkeypatch.setattr(gpu,"start",start)
        async with main_module.lane_control._FLIP_LOCK:
            assert (await gpu.flip(req()))["status"] == "noop"
        start.assert_not_called()
    asyncio.run(scenario())

def test_same_slot_concurrent_calls_do_not_race(monkeypatch):
    async def scenario():
        monkeypatch.setattr(gpu.settings,"GPU2_ENABLED",True)
        entered=asyncio.Event();release=asyncio.Event()
        async def transition(_):
            entered.set();await release.wait();return {"status":"success"}
        monkeypatch.setattr(gpu,"transition",transition)
        first=asyncio.create_task(gpu.flip(req()))
        await entered.wait()
        assert (await gpu.flip(req()))["status"] == "busy"
        release.set();await first
    asyncio.run(scenario())

@pytest.mark.parametrize("token,header,code",[("",None,503),("secret",None,401),("secret","Bearer wrong",401),("secret","secret",401)])
def test_gpu2_auth(token,header,code,monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(main_module.settings,"GPU_LANE_CONTROLLER_TOKEN",token)
    response=TestClient(main_module.app).post("/v1/gpu-slots/activate",json=req().model_dump(),headers={"Authorization":header} if header else {})
    assert response.status_code == code


def test_completed_duplicate_does_not_require_owners_to_drain(monkeypatch):
    async def check(req,*,require_drained=True):
        if require_drained:
            raise RuntimeError('authority_ownership_not_drained')
        return {}
    monkeypatch.setattr(gpu,'authority',check)
    monkeypatch.setattr(gpu,'status',AsyncMock(return_value=snapshot(active='agent-burst')))
    monkeypatch.setattr(gpu,'model_ready',AsyncMock(return_value=True))
    assert asyncio.run(gpu.transition(req()))['status']=='noop'


def test_active_upstream_prevents_restoration(monkeypatch):
    monkeypatch.setattr(gpu,'authority',AsyncMock(return_value={'can_transition':True}))
    monkeypatch.setattr(gpu,'status',AsyncMock(return_value=snapshot(active='agent-burst')))
    monkeypatch.setattr(gpu,'request',AsyncMock(return_value=[{'is_processing':True}]))
    stop=AsyncMock();monkeypatch.setattr(gpu,'stop',stop)
    result=asyncio.run(gpu.transition(req('diffusion')))
    assert result['status']=='failed' and result['error']=='burst_upstream_not_idle'
    stop.assert_not_called()


def test_authority_rechecks_thermal_eligibility_before_mutation(monkeypatch):
    monkeypatch.setattr(gpu,'request',AsyncMock(return_value={
        'operation_id':'test:1','generation':1,'desired_target':'agent-burst',
        'can_transition':True,'activation_eligible':False}))
    with pytest.raises(RuntimeError,match='activation_eligibility_suppressed'):
        asyncio.run(gpu.authority(req()))


def test_previous_rollback_is_not_evidence_for_next_operation(monkeypatch):
    monkeypatch.setattr(gpu,'authority',AsyncMock(return_value={'can_transition':True}))
    monkeypatch.setattr(gpu,'status',AsyncMock(return_value=snapshot(state='unknown')))
    gpu._state['restored']=True
    gpu._state['cold_start_seconds']=12
    result=asyncio.run(gpu.transition(req()))
    assert result['status']=='failed'
    assert 'restored' not in result and 'cold_start_seconds' not in result
