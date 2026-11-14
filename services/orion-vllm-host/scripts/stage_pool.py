#!/usr/bin/env python3
"""Render a candidate Hecate config to stdout. Never edit active config."""
import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from orion.gpu_pool.config import PoolConfig, load_pool_config


def candidate(gpus=4, backend='vllm', commissioning=False):
    if gpus not in (1, 2, 4):
        raise ValueError('Hecate candidates support one-card Qwen, two-card deep thinker or four-card GLM')
    if backend not in ('vllm', 'strata') or (backend == 'strata' and gpus != 2):
        raise ValueError('Strata comparison requires exactly two cards')
    data = load_pool_config().model_dump(mode='json', by_alias=True, exclude={'digest'})
    cards = [f'hecate-gpu{i}' for i in range(gpus)]
    for index, card in enumerate(cards):
        data['cards'][card] = dict(vram_gb=32, host='hecate', index=index, lendable=True)
    # Keep Hecate borrowing under the existing explicit lending controls; all role cards
    # must be lent before a foreign class can borrow this tensor-parallel role.
    data['roles']['agent-deep'].update(backend=backend, cards=cards)
    if commissioning:
        # Keep unvalidated hardware private while using the normal background,
        # heartbeated operator lease and heat/refusal policy. Both cards stay un-lent.
        data['roles']['agent-deep']['owner'] = 'operator_agent_deep'
    return PoolConfig.model_validate(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--backend', choices=('vllm', 'strata'), default='vllm')
    parser.add_argument('--commissioning', action='store_true', help='make un-lent Hecate private to background operator evals')
    args = parser.parse_args()
    print(f'# CANDIDATE ONLY: verify {args.gpus} V100-32GB card(s) before activation.')
    print(yaml.safe_dump(candidate(args.gpus, args.backend, args.commissioning).model_dump(mode='json', by_alias=True, exclude={'digest'}), sort_keys=False))
