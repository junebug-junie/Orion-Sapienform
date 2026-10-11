#!/usr/bin/env python3
"""Render a candidate quad-V100 Hecate config to stdout. Never edit active config."""
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from orion.gpu_pool.config import PoolConfig, load_pool_config


def candidate():
    data = load_pool_config().model_dump(mode='json', by_alias=True, exclude={'digest'})
    cards = [f'hecate-gpu{i}' for i in range(4)]
    for index, card in enumerate(cards):
        data['cards'][card] = dict(vram_gb=32, host='hecate', index=index, lendable=True)
    # Keep Hecate borrowing under the existing explicit lending controls; all four cards
    # must be lent before a foreign class can borrow this tensor-parallel role.
    data['roles']['agent-deep'].update(backend='vllm', cards=cards)
    return PoolConfig.model_validate(data)


if __name__ == '__main__':
    print('# CANDIDATE ONLY: four V100-32GB cards must be verified before activation.')
    print(yaml.safe_dump(candidate().model_dump(mode='json', by_alias=True, exclude={'digest'}), sort_keys=False))
