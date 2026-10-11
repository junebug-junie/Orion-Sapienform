#!/usr/bin/env python3
"""Render a candidate Hecate config to stdout. Never edit active config."""
import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from orion.gpu_pool.config import PoolConfig, load_pool_config


def candidate(gpus=4):
    if gpus not in (1, 4):
        raise ValueError('Hecate candidates support one-card Qwen or four-card GLM')
    data = load_pool_config().model_dump(mode='json', by_alias=True, exclude={'digest'})
    cards = [f'hecate-gpu{i}' for i in range(gpus)]
    for index, card in enumerate(cards):
        data['cards'][card] = dict(vram_gb=32, host='hecate', index=index, lendable=True)
    # Keep Hecate borrowing under the existing explicit lending controls; all role cards
    # must be lent before a foreign class can borrow this tensor-parallel role.
    data['roles']['agent-deep'].update(backend='vllm', cards=cards)
    return PoolConfig.model_validate(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    print(f'# CANDIDATE ONLY: verify {args.gpus} V100-32GB card(s) before activation.')
    print(yaml.safe_dump(candidate(args.gpus).model_dump(mode='json', by_alias=True, exclude={'digest'}), sort_keys=False))
