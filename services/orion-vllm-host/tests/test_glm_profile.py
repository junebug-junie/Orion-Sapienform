import json
from pathlib import Path

from app import main
from app.settings import Settings

ROOT = Path(__file__).resolve().parents[3]


def test_glm_profile_builds_the_pinned_forks_required_arguments(monkeypatch):
    cfg = Settings(_env_file=None, VLLM_PROFILE_NAME='glm-5.3-flash-exl3-hecate',
                   LLM_PROFILES_CONFIG_PATH=ROOT/'config/llm_profiles.yaml')
    monkeypatch.setattr(main, 'settings', cfg)
    cmd, env = main.build_vllm_command_and_env()
    def arg(flag): return cmd[cmd.index(flag)+1]
    assert arg('--middleware') == 'app.discovery.pool_server_info'
    assert 'VLLM_SERVER_DEV_MODE' not in env
    assert arg('--model') == '/models/GLM-5.3-Flash-EXL3'
    assert arg('--served-model-name') == 'glm-5.3-flash'
    assert arg('--dtype') == 'half'
    assert arg('--tensor-parallel-size') == '4'
    assert arg('--max-num-seqs') == '1'
    assert arg('--kv-cache-dtype') == 'fp8_e4m3'
    assert arg('--max-model-len') == '131072'
    assert '--trust-remote-code' in cmd and '--enable-auto-tool-choice' in cmd
    assert json.loads(arg('--speculative-config'))['num_speculative_tokens'] == 3
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    assert env['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] == '0'
    assert env['VLLM_USE_V2_MODEL_RUNNER'] == '1'


def test_empty_announcement_port_is_disabled():
    assert Settings(_env_file=None, LLM_ANNOUNCE_PORT='').llm_announce_port is None
