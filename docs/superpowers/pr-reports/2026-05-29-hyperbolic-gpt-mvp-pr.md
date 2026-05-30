# PR: feat(substrate) — Hyperbolic GPT attention MVP

**Branch:** `feat/hyperbolic-gpt-mvp`  
**Base:** `main`  
**Compare:** https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/hyperbolic-gpt-mvp

## Summary

Adds a self-contained research experiment at `orion/substrate/experiments/hyperbolic_gpt/`: a tiny decoder-only GPT where **Poincaré distance is subtracted from attention logits inside `HyperbolicCausalSelfAttention.forward`** (before causal mask and softmax), not as post-hoc analysis.

- Hand-rolled Poincaré ball math (`hyperbolic.py`), fp32 geo ops in attention, fp16 AMP elsewhere
- Training on TinyStories or local text (GPT-2 tokenizer)
- Single-node **DDP** via `torchrun` (rank-0 checkpoints, `model.module.state_dict()`)
- Smoke test, generation CLI, shell scripts
- No Docker, bus, or service changes

## Files changed

| File | Change |
|------|--------|
| `docs/superpowers/plans/2026-05-29-hyperbolic-gpt-mvp.md` | Implementation plan |
| `docs/superpowers/pr-reports/2026-05-29-hyperbolic-gpt-mvp-pr.md` | This report |
| `orion/substrate/experiments/__init__.py` | Package marker |
| `orion/substrate/experiments/hyperbolic_gpt/config.py` | `HyperbolicGPTConfig` |
| `orion/substrate/experiments/hyperbolic_gpt/hyperbolic.py` | Poincaré primitives + pairwise distances |
| `orion/substrate/experiments/hyperbolic_gpt/model.py` | `HyperbolicGPT` + hyperbolic causal attention |
| `orion/substrate/experiments/hyperbolic_gpt/data.py` | Tokenizer, TinyStories/text, sharding |
| `orion/substrate/experiments/hyperbolic_gpt/train.py` | AMP, DDP, checkpoints |
| `orion/substrate/experiments/hyperbolic_gpt/generate.py` | Sampling from checkpoint |
| `orion/substrate/experiments/hyperbolic_gpt/smoke_test.py` | Forward/backward/generate sanity |
| `orion/substrate/experiments/hyperbolic_gpt/README.md` | Usage, memory/DDP notes |
| `orion/substrate/experiments/hyperbolic_gpt/requirements.txt` | torch, transformers, datasets, … |
| `orion/substrate/experiments/hyperbolic_gpt/scripts/*.sh` | smoke, train, generate wrappers |

**Not changed:** `services/*`, docker-compose, `orion/bus/channels.yaml`, `orion/schemas/registry.py`, root `requirements.txt`.

## Attention modification

```text
attn_logits = (Q @ K^T / sqrt(d)) - softplus(λ) * d_Poincaré(expmap0(q_geo), expmap0(k_geo))
attn = softmax(causal_mask(attn_logits)) @ V
```

## Code review

- **Verdict:** APPROVED (research MVP)
- **Smoke test:** `SMOKE TEST PASSED` (CPU; host P100/P4 incompatible with default torch CUDA wheels)
- **Follow-ups applied:** `--seed`, `--max_tokens`, README memory/DDP eval notes, batch 16 in train script, safer `torch.load`, eval shard + optimizer-step semantics from prior review

## Test plan

- [x] `PYTHONPATH=. CUDA_VISIBLE_DEVICES= python3 orion/substrate/experiments/hyperbolic_gpt/smoke_test.py`
- [x] CPU `train.py` dry-run (2 optimizer steps on local text)
- [ ] Optional: 10k-step DDP TinyStories on CUDA GPUs with compatible PyTorch (e.g. V100+)

## Commands

### Setup

```bash
cd /path/to/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
pip install -r orion/substrate/experiments/hyperbolic_gpt/requirements.txt
export PYTHONPATH=.
```

### Smoke test

```bash
CUDA_VISIBLE_DEVICES= python3 orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
```

### DDP TinyStories (2 GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset tinystories \
  --max_docs 50000 \
  --max_tokens 5000000 \
  --out_dir ./runs/tinystories_hypgpt_4l_256d \
  --max_steps 10000 \
  --batch_size 16 \
  --block_size 256 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --lr 3e-4 \
  --eval_interval 250 \
  --save_interval 1000 \
  --amp
```

Or: `bash orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh`

### Single-GPU

```bash
python3 orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset tinystories --device cuda --amp
```

### Generate

```bash
python3 orion/substrate/experiments/hyperbolic_gpt/generate.py \
  --checkpoint ./runs/tinystories_hypgpt_4l_256d \
  --prompt "Once upon a time" --max_new_tokens 120 --temperature 0.9 --top_k 40
```

## Known limitations

- Hyperbolic pairwise distance is **O(B·H·T²·D)** memory in attention — tune batch size for 16GB GPUs.
- TinyStories tokens are materialized per process; use `--max_docs` / `--max_tokens`.
- DDP eval loss is a sharded estimate, not exact corpus NLL.
- Research code; no Euclidean baseline in this PR.
