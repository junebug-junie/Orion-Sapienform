import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HUB = Path(__file__).resolve().parents[2]


def test_plan_boundary_failure_and_native_rejection(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HUB))
    from scripts import retro_planner
    spec = importlib.util.spec_from_file_location('plan_routes', HUB/'scripts/retro_routes.py')
    routes = importlib.util.module_from_spec(spec); spec.loader.exec_module(routes)
    (tmp_path/'hub.token').write_text('x'*48)
    cfg = SimpleNamespace(HUB_RETRO_ENABLED=True, HUB_RETRO_COGNITION_ENABLED=False,
        HUB_RETRO_ORIGIN='http://testserver', HUB_RETRO_AUTH_DIR=str(tmp_path), HUB_RETRO_SOCKET='/unused')
    state = dict(generation='g'*36, scripted=[dict(port=0,epoch=2,cognition=dict(attempt=dict(
        attempt_id='2:1', observation={'status':'unknown'}, supported_goals=['wait'])))])
    calls=[]
    class Response:
        def __init__(self, data, status=200): self.data=data; self.status=status
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def json(self): return self.data
    class Client:
        def __init__(self,**kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        def get(self,*args,**kwargs): return Response(state)
        def post(self,url,**kwargs):
            calls.append(kwargs['json'])
            return Response({'detail':'goal_requires_active_controller'},409)
    monkeypatch.setattr(routes.aiohttp,'ClientSession',Client)
    async def choose(config, attempt):
        return dict(goal='wait',rationale='unknown observation',planner_id='fixture')
    monkeypatch.setattr(retro_planner,'choose',choose)
    app=FastAPI();app.include_router(routes.create_router(cfg))
    body=dict(connection_id='c'*36,generation='g'*36,port=0,expected_epoch=2)
    path='/api/retro/sessions/'+'a'*36+'/plan'
    with TestClient(app) as c:
        h={'Origin':'http://testserver'}
        assert c.post(path,json=body).status_code==403
        assert c.post(path,json=body,headers=h).status_code==503
        cfg.HUB_RETRO_COGNITION_ENABLED=True
        assert c.post(path,json=body|{'port':True},headers=h).status_code==422
        assert c.post(path,json=body|{'expected_epoch':1},headers=h).status_code==409
        assert not calls
        assert c.post(path,json=body,headers=h).status_code==409
        assert calls[0]['attempt_id']=='2:1'
        async def unavailable(*args): raise TimeoutError()
        monkeypatch.setattr(retro_planner,'choose',unavailable)
        monkeypatch.setattr(routes.time, 'monotonic', lambda: 10**12)
        assert c.post(path,json=body,headers=h).status_code==503
        assert len(calls)==1  # no fallback goal on planner failure


def test_choice_rejects_arbitrary_goals_and_empty_cognition(monkeypatch):
    monkeypatch.syspath_prepend(str(HUB))
    from scripts.retro_planner import Choice
    from pydantic import ValidationError
    for body in ({'goal':'reach_exit','rationale':'yes'}, {'goal':'wait','rationale':''},
                 {'goal':'wait','rationale':'unknown','tool':'shell'}):
        with pytest.raises(ValidationError): Choice(**body)
