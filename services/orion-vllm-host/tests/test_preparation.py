from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / 'services/orion-vllm-host'


def test_candidate_allocates_all_four_cards_and_preserves_circe():
    from orion.gpu_pool.config import load_pool_config
    cfg = runpy.run_path(str(SERVICE/'scripts/stage_pool.py'))['candidate']()
    live = load_pool_config()
    assert cfg.roles['agent-deep'].backend == 'vllm'
    assert cfg.roles['agent-deep'].cards == [f'hecate-gpu{i}' for i in range(4)]
    assert cfg.url('agent-deep') == 'http://100.87.202.68:8021'
    for role in live.roles:
        if role != 'agent-deep':
            assert live.roles[role] == cfg.roles[role]
    assert live.classes == cfg.classes and live.routes == cfg.routes
    assert live.roles['agent-deep'].backend == 'llamacpp'


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
