import copy
import json
import itertools
from pathlib import Path
import runpy

import pytest

from app import main
from app.deep_budget import budget_error
from app.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
EVAL = runpy.run_path(str(ROOT / 'services/orion-vllm-host/evals/deep_context.py'))


def test_launch_has_two_cards_full_context_and_one_sequence(monkeypatch):
    monkeypatch.setattr(main, 'settings', Settings(_env_file=None, VLLM_PROFILE_NAME='deep.thinker.v1',
                        LLM_PROFILES_CONFIG_PATH=ROOT / 'config/llm_profiles.yaml'))
    cmd, env = main.build_vllm_command_and_env()
    for flag, value in {'--tensor-parallel-size': '2', '--max-model-len': '262144',
                        '--max-num-seqs': '1', '--attention-backend': 'FLASH_ATTN_V100',
                        '--kv-cache-dtype': 'fp8_e4m3', '--dtype': 'half'}.items():
        assert cmd[cmd.index(flag) + 1] == value
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert '--enable-prefix-caching' in cmd and '--enable-chunked-prefill' in cmd
    assert '--speculative-config' not in cmd
    assert json.loads(cmd[cmd.index('--override-generation-config') + 1]) == {'max_new_tokens': 32768}


def test_budget_never_silently_shrinks_prompt_or_reasoning_reserve():
    assert budget_error({'max_tokens': 32768}, 229376, 262144) is None
    assert budget_error({'max_tokens': 16384}, 245760, 262144) is None
    assert budget_error({'max_tokens': 256}, 245761, 262144)
    assert budget_error({'max_tokens': 32768}, 229377, 262144)
    assert budget_error({'truncate_prompt_tokens': 132000}, 132000, 262144)
    assert budget_error({}, 132000, 8192)
    for value in (True, 0, 32769, '32768', 1.5):
        assert budget_error({'max_tokens': value}, 8192, 262144)


def test_evaluation_requires_actual_tokens_correct_model_and_completed_reasoning():
    expected = {'early': 'a', 'middle': 'b', 'late': 'c', 'total': 37}
    good = {'model': 'deep.thinker.v1', 'content': json.dumps(expected), 'finish_reason': 'stop',
            'usage': {'prompt_tokens': 132000, 'completion_tokens': 24}}
    def check(result):
        return EVAL['judge'](result, expected=expected, model='deep.thinker.v1',
                             prompt_tokens=132000, output_budget=32768)
    assert not check(good)
    for key, value in [('model', 'other'), ('content', ''), ('finish_reason', 'length'),
                       ('usage', {'prompt_tokens': 131072, 'completion_tokens': 24}),
                       ('usage', None), ('content', '{"total":37}')]:
        bad = copy.deepcopy(good)
        bad[key] = value
        assert check(bad)


def test_two_card_candidate_preserves_circe_and_routes():
    from orion.gpu_pool.config import load_pool_config
    live = load_pool_config()
    cfg = runpy.run_path(str(ROOT / 'services/orion-vllm-host/scripts/stage_pool.py'))['candidate'](2)
    assert cfg.roles['agent-deep'].cards == ['hecate-gpu0', 'hecate-gpu1']
    assert cfg.classes == live.classes and cfg.routes == live.routes
    for role in live.roles:
        if role != 'agent-deep':
            assert cfg.roles[role] == live.roles[role]


def test_prompt_builder_hits_exact_tokens_and_changes_trial_prefix():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"].split() + ["assistant", "thinking"]
    a, expected = EVAL['build_messages'](Tokenizer(), 8192, 123)
    b, _ = EVAL['build_messages'](Tokenizer(), 8192, 124)
    assert len(Tokenizer().apply_chat_template(a)) == 8192
    assert a != b
    assert all(expected[key] in a[0]['content'] for key in ('early', 'middle', 'late'))


def test_memory_estimate_rejects_unaccounted_kv_head_replication():
    module = runpy.run_path(str(ROOT / 'services/orion-vllm-host/scripts/deep_memory.py'))
    cfg = json.loads((ROOT / 'docs/bench/2026-11-14-hecate-deep/model-config.json').read_text())
    row = module['budget'](cfg, 20559281336, 262144)
    assert row['components_gib_total']['attention_kv'] == 8
    assert row['status'] == 'ESTIMATE'
    with pytest.raises(ValueError, match='replication'):
        module['budget'](cfg, 20559281336, 262144, tp=3)


def test_planning_gold_is_optimal_over_all_topological_orders():
    rows = [json.loads(line) for line in (ROOT / 'services/orion-vllm-host/evals/deep_tasks.jsonl').read_text().splitlines()]
    gold = next(row['expected'] for row in rows if row['id'] == 'bounded-plan')
    tasks = {'A': ('cpu', 3, ()), 'B': ('gpu', 8, ('A',)), 'C': ('cpu', 5, ('A',)),
             'D': ('gpu', 4, ('B', 'C')), 'E': ('cpu', 2, ('D',)), 'F': ('cpu', 7, ('B',))}
    scores = []
    for order in itertools.permutations(tasks):
        end, free = {}, {'cpu': 0, 'gpu': 0}
        for name in order:
            resource, duration, deps = tasks[name]
            if not all(d in end for d in deps):
                break
            end[name] = max(free[resource], max((end[d] for d in deps), default=0)) + duration
            free[resource] = end[name]
        else:
            scores.append((max(end.values()), end['E']))
    assert (gold['makespan'], gold['E_start'] + 2) == min(scores)
    assert gold == {'makespan': 20, 'B_start': 3, 'C_start': 3, 'D_start': 11, 'E_start': 18, 'F_start': 11}
