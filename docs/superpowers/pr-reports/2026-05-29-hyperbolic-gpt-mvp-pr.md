# PR: feat(substrate) — Hyperbolic GPT attention MVP

**Branch:** `feat/hyperbolic-gpt-mvp`  
**Base:** `main`  
**Worktree:** `.worktrees/feat-hyperbolic-gpt-mvp`

## Summary

Adds a self-contained research experiment at `orion/substrate/experiments/hyperbolic_gpt/`: a tiny decoder-only GPT where **Poincaré distance is subtracted from attention logits inside `HyperbolicCausalSelfAttention.forward`** (before causal mask and softmax), not as post-hoc analysis.

Includes hand-rolled Poincaré ball math, GPT-2 tokenizer + TinyStories/text training, AMP, single-node **DDP via torchrun**, smoke test, and generation CLI. No Docker, bus, or service wiring.

## Files changed

| File | Change |
|------|--------|
| `docs/superpowers/plans/2026-05-29-hyperbolic-gpt-mvp.md` | Implementation plan |
| `orion/substrate/experiments/__init__.py` | Package marker |
| `orion/substrate/experiments/hyperbolic_gpt/*` | MVP module (config, hyperbolic, model, data, train, generate, smoke_test, README, scripts) |

**Not changed:** services, docker-compose, `orion/bus/channels.yaml`, `orion/schemas/registry.py`, root `requirements.txt`.

## Core attention modification

```text
attn_logits = (Q @ K^T / sqrt(d)) - softplus(λ) * d_Poincaré(expmap0(q_geo), expmap0(k_geo))
attn = softmax(mask(attn_logits)) @ V
```

Hyperbolic ops run in fp32 inside the attention block; outer training may use fp16 AMP.

## Verification

```bash
cd .worktrees/feat-hyperbolic-gpt-mvp
pip install -r orion/substrate/experiments/hyperbolic_gpt/requirements.txt
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES= python3 orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
# => SMOKE TEST PASSED

python3 orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset text --text_path /path/to/corpus.txt \
  --max_steps 2 --device cpu --no-amp ...
# => training complete
```

**Note:** Host GPUs (P100/P4) may be incompatible with default PyTorch CUDA wheels; smoke test auto-falls back to CPU when CUDA init fails.

## Commands

### Smoke test

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES= python3 orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
```

### Single-GPU / CPU training

```bash
export PYTHONPATH=.
python3 orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset tinystories --device cuda --amp
```

### DDP TinyStories (2 GPU)

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset tinystories \
  --max_docs 50000 \
  --out_dir ./runs/tinystories_hypgpt_4l_256d \
  --max_steps 10000 \
  --batch_size 32 \
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

## Test plan

- [x] `smoke_test.py` — forward, backward, short generate, no NaNs (CPU)
- [x] `train.py` — 2-step CPU dry-run on local text
- [ ] Optional: full 10k-step DDP TinyStories on CUDA-capable GPUs (V100+ with matching torch build)

## Known limitations

- TinyStories still materializes tokens per process; `--max_docs` caps RAM (default 50k in shell script).
- Research code; no Euclidean baseline or benchmark harness in this PR.
