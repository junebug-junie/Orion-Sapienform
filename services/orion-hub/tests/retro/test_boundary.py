import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

spec=importlib.util.spec_from_file_location('retro_routes',Path(__file__).resolve().parents[2]/'scripts/retro_routes.py')
routes=importlib.util.module_from_spec(spec);spec.loader.exec_module(routes)


def test_network_boundary_without_access_code(tmp_path):
    (tmp_path/'hub.token').write_text('y'*48)
    cfg=SimpleNamespace(HUB_RETRO_ENABLED=True,HUB_RETRO_ORIGIN='http://testserver',HUB_RETRO_AUTH_DIR=str(tmp_path),HUB_RETRO_SOCKET='/nonexistent')
    app=FastAPI();app.include_router(routes.create_router(cfg))
    with TestClient(app) as c:
        # No browser credential file or cookie: reaches upstream (unavailable in this test).
        assert c.get('/api/retro/games').status_code==503
        assert c.post('/api/retro/sessions',json={}).status_code==403
        assert c.post('/api/retro/sessions',json={},headers={'Origin':'https://evil.example'}).status_code==403
        good={'Origin':'http://testserver'}
        assert c.post('/api/retro/sessions',json={},headers=good).status_code==503
        assert c.post('/api/retro/login',json={},headers=good).status_code==404
        assert c.post('/api/retro/logout',json={},headers=good).status_code==404
        assert c.get('/api/retro/arbitrary').status_code==404
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect('/api/retro/sessions/'+'a'*36+'/input',headers={'Origin':'https://evil.example'}):pass
        html=c.get('/retro').text
        assert '/static/js/retro-panel.js' in html
        assert 'id="login"' not in html and 'id="code"' not in html
        assert 'y'*48 not in html
        cfg.HUB_RETRO_ENABLED=False
        assert c.get('/api/retro/games').status_code==503


def test_gateway_process_serves_only_retro_with_the_same_boundary(tmp_path, monkeypatch):
    # The dedicated relay must enforce Hub's origin rule and expose nothing but Retro.
    import sys
    hub=Path(__file__).resolve().parents[2]
    (tmp_path/'hub.token').write_text('y'*48)
    for k,v in dict(HUB_RETRO_ENABLED='true',HUB_RETRO_ORIGIN='http://testserver',HUB_RETRO_AUTH_DIR=str(tmp_path),HUB_RETRO_SOCKET='/nonexistent').items():
        monkeypatch.setenv(k,v)
    monkeypatch.syspath_prepend(str(hub))
    for name in [m for m in sys.modules if m=='app' or m.startswith(('app.','scripts'))]:
        monkeypatch.delitem(sys.modules,name)
    from scripts.retro_gateway import app
    paths={r.path for r in app.routes}
    assert paths=={'/retro','/api/retro/{path:path}','/api/retro/sessions/{sid}/{channel}','/api/retro-gateway/health'}
    with TestClient(app) as c:
        assert c.get('/api/retro-gateway/health').json()=={'status':'ok'}
        assert c.post('/api/retro/sessions',json={},headers={'Origin':'https://evil.example'}).status_code==403
        assert c.post('/api/retro/sessions',json={},headers={'Origin':'http://testserver'}).status_code==503


def test_gateway_compose_service_is_loopback_and_skips_the_bus_wait():
    import yaml
    compose=yaml.safe_load((Path(__file__).resolve().parents[2]/'docker-compose.yml').read_text())
    gw=compose['services']['hub-retro-gateway']
    assert gw['entrypoint']==[] and gw['command'][:2]==['uvicorn','scripts.retro_gateway:app']
    assert gw['command'][gw['command'].index('--host')+1]=='127.0.0.1'
    assert gw['volumes']==compose['services']['hub-app']['volumes'][:2]  # socket + auth only
