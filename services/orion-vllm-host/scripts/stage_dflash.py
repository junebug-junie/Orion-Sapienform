#!/usr/bin/env python3
"""Render an optional DFlash2 profile override; never change default routing or start it."""
from pathlib import Path
import yaml


def candidate():
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((root / 'config/llm_profiles.yaml').read_text())
    cfg['profiles']['deep.thinker.v1']['vllm']['speculative_config'] = {
        'method': 'dflash', 'model': '/models/Qwen3.8-27B-DFlash2',
        'num_speculative_tokens': 7, 'draft_tensor_parallel_size': 2,
        'attention_backend': 'FLASH_ATTN_V100', 'kv_cache_dtype': 'auto',
    }
    return cfg


if __name__ == '__main__':
    print(yaml.safe_dump(candidate(), sort_keys=False))
