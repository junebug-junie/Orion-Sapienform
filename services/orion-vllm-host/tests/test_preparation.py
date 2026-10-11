from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / 'services/orion-vllm-host'


def test_candidate_allocates_all_four_cards_and_preserves_circe():
    from orion.gpu_pool.config import load_pool_config
    live = load_pool_config()
    source = ROOT / 'config/gpu_pool.yaml'
    before = source.read_bytes()
    cfg = runpy.run_path(str(SERVICE/'scripts/stage_pool.py'))['candidate']()
    assert cfg.roles['agent-deep'].backend == 'vllm'
    assert cfg.roles['agent-deep'].cards == [f'hecate-gpu{i}' for i in range(4)]
    assert cfg.url('agent-deep') == 'http://100.87.202.68:8021'
    for role in live.roles:
        if role != 'agent-deep':
            assert live.roles[role] == cfg.roles[role]
    assert live.classes == cfg.classes and live.routes == cfg.routes
    assert load_pool_config() == live  # Rendering never mutates the source config.
    assert source.read_bytes() == before


def test_smoke_rejects_empty_reasoning_only_or_wrong_model(monkeypatch):
    module = runpy.run_path(str(SERVICE/'evals/smoke_glm.py'))
    evaluate = module['evaluate']
    def reply(*args, **kwargs):
        return {'model':'glm-5.3-flash','choices':[{'message':{'content':None,'reasoning_content':'still thinking'}}]}
    monkeypatch.setitem(evaluate.__globals__, 'chat', reply)
    assert not any(x['passed'] for x in evaluate('http://unused', 'agent-deep', 'glm-5.3-flash'))
    def wrong(*args, **kwargs):
        return {'model':'wrong-model','choices':[{'message':{'content':'391'}}]}
    monkeypatch.setitem(evaluate.__globals__, 'chat', wrong)
    assert not any(x['passed'] for x in evaluate('http://unused', 'agent-deep', 'glm-5.3-flash'))


def test_single_card_candidate_preserves_other_roles_and_routing():
    from orion.gpu_pool.config import load_pool_config
    live = load_pool_config()
    candidate = runpy.run_path(str(SERVICE/'scripts/stage_pool.py'))['candidate']
    cfg = candidate(gpus=1)
    assert cfg.roles['agent-deep'].backend == 'vllm'
    assert cfg.roles['agent-deep'].cards == ['hecate-gpu0']
    assert cfg.cards == live.cards
    assert cfg.classes == live.classes and cfg.routes == live.routes
    for role in live.roles:
        if role != 'agent-deep':
            assert cfg.roles[role] == live.roles[role]
    import pytest
    with pytest.raises(ValueError):
        candidate(gpus=2)


def test_hecate_wrapper_rejects_other_host_before_docker(tmp_path):
    import os
    import subprocess
    fakebin = tmp_path / 'bin'
    fakebin.mkdir()
    hostname = fakebin / 'hostname'
    hostname.write_text('#!/bin/sh\necho athena\n')
    hostname.chmod(0o755)
    result = subprocess.run([str(SERVICE/'scripts/hecate.sh'), 'up', '-d'],
                            env={**os.environ, 'PATH':str(fakebin)+':'+os.environ['PATH']},
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'must run on Hecate' in result.stderr


def test_hecate_wrapper_rejects_wrong_model_mount(tmp_path):
    import os
    import subprocess
    fakebin = tmp_path / 'bin'
    fakebin.mkdir()
    for name, value in [('hostname', 'hecate'), ('findmnt', '/dev/sda1')]:
        command = fakebin / name
        command.write_text('#!/bin/sh\necho '+value+'\n')
        command.chmod(0o755)
    result = subprocess.run([str(SERVICE/'scripts/hecate.sh'), 'up', '-d'],
                            env={**os.environ, 'PATH':str(fakebin)+':'+os.environ['PATH']},
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'NVMe is not mounted' in result.stderr


def test_hecate_wrapper_allows_shutdown_with_broken_storage(tmp_path):
    import os
    import subprocess
    root = tmp_path / 'repo'
    wrapper = root / 'services/orion-vllm-host/scripts/hecate.sh'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_bytes((SERVICE / 'scripts/hecate.sh').read_bytes())
    wrapper.chmod(0o755)
    safe = root / 'scripts/safe_docker_build.sh'
    safe.parent.mkdir()
    safe.write_text('#!/bin/sh\necho safe-wrapper-reached\n')
    safe.chmod(0o755)
    fakebin = tmp_path / 'bin'
    fakebin.mkdir()
    for name, body in [('hostname', 'echo hecate'), ('findmnt', 'exit 99'), ('python3', 'exit 99')]:
        command = fakebin / name
        command.write_text('#!/bin/sh\n' + body + '\n')
        command.chmod(0o755)
    for verb in ('stop', 'down', 'logs', 'ps', 'config'):
        result = subprocess.run([str(wrapper), verb],
                                env={**os.environ, 'PATH': str(fakebin) + ':' + os.environ['PATH']},
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'safe-wrapper-reached' in result.stdout
