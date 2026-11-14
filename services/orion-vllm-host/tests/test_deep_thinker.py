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


def test_dflash_candidate_changes_only_speculation_and_keeps_full_context():
    import yaml
    original = yaml.safe_load((ROOT / 'config/llm_profiles.yaml').read_text())
    candidate = runpy.run_path(str(ROOT / 'services/orion-vllm-host/scripts/stage_dflash.py'))['candidate']()
    spec = candidate['profiles']['deep.thinker.v1']['vllm'].pop('speculative_config')
    assert candidate == original
    assert spec['method'] == 'dflash' and spec['num_speculative_tokens'] == 7
    assert spec['draft_tensor_parallel_size'] == 2 and spec['kv_cache_dtype'] == 'auto'


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


def test_strata_candidate_changes_only_hecate_backend():
    stage = runpy.run_path(str(ROOT / 'services/orion-vllm-host/scripts/stage_pool.py'))['candidate']
    baseline, challenger = stage(2), stage(2, 'strata')
    assert challenger.roles['agent-deep'].backend == 'strata'
    challenger.roles['agent-deep'].backend = 'vllm'
    assert challenger == baseline
    with pytest.raises(ValueError):
        stage(1, 'strata')
    private = stage(2, 'vllm', commissioning=True)
    assert private.roles['agent-deep'].owner == ['operator_agent_deep']
    private.roles['agent-deep'].owner = baseline.roles['agent-deep'].owner
    assert private == baseline


def test_commissioning_keeps_unvalidated_worker_private_and_heat_gated():
    from orion.gpu_pool.scheduler import schedule, RoleLive, CardLive
    from orion.gpu_pool.tests.test_scheduler import lease, T0, grants
    stage = runpy.run_path(str(ROOT / 'services/orion-vllm-host/scripts/stage_pool.py'))['candidate']
    cfg = stage(2, 'vllm', commissioning=True)
    roles = {name: RoleLive(name, name == 'agent-deep', 1, 262144, False) for name in cfg.roles}
    cards = {name: CardLive(name) for name in cfg.cards}
    for work_class in ('agent-deep', 'agent', 'metacog', 'fast', 'operator', 'operator_agent_deep'):
        queued = lease(work_class, lease_id=work_class, priority='background', kind='hold')
        actual = grants(schedule(cfg, roles, cards, [queued], T0))
        assert actual == ({work_class: 'agent-deep'} if work_class == 'operator_agent_deep' else {})
    queued = lease('operator_agent_deep', lease_id='op', priority='background', kind='hold')
    assert grants(schedule(cfg, roles, cards, [queued], T0, shed={'background': 'cabinet_hot'})) == {}


def test_prompt_builder_hits_exact_tokens_and_changes_trial_prefix():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            ids = messages[0]["content"].split() + ["assistant", "thinking"]
            return ids if kwargs.get('return_dict') is False else {'input_ids': ids, 'attention_mask': [1] * len(ids)}
    a, expected = EVAL['build_messages'](Tokenizer(), 8192, 123)
    b, _ = EVAL['build_messages'](Tokenizer(), 8192, 124)
    assert len(Tokenizer().apply_chat_template(a, return_dict=False)) == 8192
    assert a != b
    assert all(expected[key] in a[0]['content'] for key in ('early', 'middle', 'late'))


def test_hard_context_gold_includes_reassignment_decoy_conversion_and_date_updates():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]['content'].split()
    messages, expected = EVAL['build_messages'](Tokenizer(), 8192, 123, 'hard')
    assert len(Tokenizer().apply_chat_template(messages)) == 8192
    # Reference case: 2799 lb * .4536 + 686 kg + (686-67) kg = 2574.6264 kg;
    # Monday moved 3+3 days is Sunday; the replacement has badge 8147, not 5016.
    assert expected == {'current_badge': 8147, 'other_badge': 3683, 'total_kg': 2575,
                        'review_day': 'Sunday', 'previous_keeper': 'Mirela Cask'}
    assert '2799 pounds' in messages[0]['content'] and 'badge 5016' in messages[0]['content']


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
