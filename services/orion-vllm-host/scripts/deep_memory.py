#!/usr/bin/env python3
"""Analytical lower bound plus explicit reserves, NOT a measured memory capacity."""
import argparse
import json

GIB = 1024**3


def budget(config, weight_bytes, tokens, kv_bytes=1, tp=2):
    c = config.get('text_config', config)
    attention_layers = c['layer_types'].count('full_attention')
    linear_layers = c['layer_types'].count('linear_attention')
    # Sharded GQA (must divide TP); K and V both reside in HBM.
    if c['num_key_value_heads'] % tp:
        raise ValueError('KV heads do not divide TP: replication needs a different budget')
    kv = 2 * attention_layers * c['num_key_value_heads'] * c['head_dim'] * tokens * kv_bytes
    # FP32 GDN matrices, including snapshots at the selected retention interval.
    state = linear_layers * c['linear_num_value_heads'] * c['linear_key_head_dim'] * c['linear_value_head_dim'] * 4
    states = state * (1 + (tokens + 8191) // 8192)
    components = {'checkpoint_weights_and_quant_metadata': weight_bytes / GIB,
                  'attention_kv': kv / GIB, 'gdn_states_with_retention_estimate': states / GIB,
                  'cuda_runtime_graphs_reserve': 4.0, 'attention_workspace_reserve': 2.0,
                  'temporary_prefill_reserve': 4.0, 'fragmentation_safety_reserve': 6.0,
                  'speculative_overhead': 0.0}
    return {'tokens': tokens, 'status': 'ESTIMATE', 'components_gib_total': components,
            'total_gib': sum(components.values()), 'ideal_per_rank_gib': sum(components.values()) / tp,
            'caveats': 'Weight replication/repacking and cache allocator padding require live profiling; draft disabled.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config'); p.add_argument('index')
    args = p.parse_args()
    config = json.load(open(args.config)); weights = json.load(open(args.index))['metadata']['total_size']
    print(json.dumps([budget(config, weights, n) for n in (132000, 196608, 262144)], indent=2))
