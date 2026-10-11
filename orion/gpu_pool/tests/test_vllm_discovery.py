"""No vLLM capacity exists until live engine facts agree with the announcement."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from orion.gpu_pool.config import PoolConfig, load_pool_config
from orion.gpu_pool.discovery import Probe, load_profiles, resolve_roles
from orion.gpu_pool.route_view import build_route_view
from orion.gpu_pool.scheduler import CardLive
from orion.schemas.gpu_pool import LlmWorkerAnnounceV1

PROFILE = 'glm-5.3-flash-exl3-hecate'
NOW = datetime.now(timezone.utc)


def configuration():
    data = load_pool_config().model_dump(by_alias=True)
    cards = [f'hecate-gpu{i}' for i in range(4)]
    for i, card in enumerate(cards):
        data['cards'][card] = dict(vram_gb=32, host='hecate', index=i)
    data['roles']['agent-deep'].update(backend='vllm', cards=cards)
    return PoolConfig.model_validate(data)


def engine():
    return {'models': {'data': [{'id': 'glm-5.3-flash'}]},
            'server_info': {'vllm_config': {
                'model_config': {'model': '/models/GLM-5.3-Flash-EXL3', 'max_model_len': 131072},
                'scheduler_config': {'max_num_seqs': 1},
                'parallel_config': {'tensor_parallel_size': 4, 'pipeline_parallel_size': 1, 'data_parallel_size': 1}}}}


def announce():
    return LlmWorkerAnnounceV1(host='hecate', role='agent-deep', port=8021,
                              profile_name=PROFILE, cuda_visible_devices='0,1,2,3', announced_at=NOW)


def resolve(payload=None, ann=None, backend='vllm', ok=True):
    cfg = configuration()
    rows, live, _ = resolve_roles(cfg, load_profiles(), {'agent-deep': ann or announce()},
                                  {'agent-deep': Probe(ok, payload or engine(), backend=backend)},
                                  {c: CardLive(c) for c in cfg.cards}, NOW)
    return next(r for r in rows if r.role == 'agent-deep'), live['agent-deep']


def test_live_engine_capacity_alias_and_path_are_preserved():
    row, live = resolve()
    assert row.status == 'confirmed' and live.healthy
    assert (row.model_file, row.model_path, row.slots, row.ctx_per_slot) == (
        'glm-5.3-flash', '/models/GLM-5.3-Flash-EXL3', 1, 131072)
    # Capability not claimed from a model card; images remain UNVERIFIED.
    assert row.vision is None


@pytest.mark.parametrize('field,value', [
    ('model', '/models/wrong'), ('max_model_len', 0), ('max_model_len', '131072'),
])
def test_bad_model_facts_refuse_grants(field, value):
    data = engine()
    data['server_info']['vllm_config']['model_config'][field] = value
    assert not resolve(data)[1].healthy


@pytest.mark.parametrize('section,field,value', [
    ('scheduler_config', 'max_num_seqs', 0), ('scheduler_config', 'max_num_seqs', True),
    ('parallel_config', 'tensor_parallel_size', 1),
    ('parallel_config', 'pipeline_parallel_size', 2), ('parallel_config', 'data_parallel_size', 2),
])
def test_invalid_capacity_or_partial_allocation_refuses(section, field, value):
    data = engine()
    data['server_info']['vllm_config'][section][field] = value
    assert not resolve(data)[1].healthy


@pytest.mark.parametrize('change', [dict(host='circe'), dict(port=8000),
                                  dict(cuda_visible_devices='0'), dict(profile_name='missing')])
def test_wrong_announcement_refuses(change):
    assert not resolve(ann=announce().model_copy(update=change))[1].healthy


def test_stale_down_wrong_alias_and_wrong_probe_refuse():
    row, live = resolve(ann=announce().model_copy(update={'announced_at': NOW-timedelta(minutes=3)}))
    assert row.status == 'silent' and not live.healthy
    assert not resolve(ok=False)[1].healthy
    assert not resolve(backend='llamacpp')[1].healthy
    data = engine()
    data['models']['data'][0]['id'] = 'wrong-model'
    assert not resolve(data)[1].healthy
    data['server_info'] = {'vllm_config': 'text format'}
    assert not resolve(data)[1].healthy


def test_legacy_role_serialization_is_unchanged_and_active_config_stays_llamacpp():
    cfg = load_pool_config()
    assert all(r.backend == 'llamacpp' for r in cfg.roles.values())
    assert all('backend' not in r for r in cfg.model_dump()['roles'].values())
    assert configuration().model_dump()['roles']['agent-deep']['backend'] == 'vllm'


def test_routes_report_vllm_without_claiming_health():
    cfg = configuration()
    route = next(r for r in build_route_view(None, cfg)['routes'] if r['id'] == 'agent-deep')
    assert route['backend'] == 'vllm' and route['status'] == 'unknown'
